# 导入所需依赖库
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------------------------------------
# 1. 读取数据集
# ------------------------------------------------------------------------------
# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建数据文件的路径
data_path = os.path.join(script_dir, 'car_evaluation.csv')
df = pd.read_csv(data_path, header=None)
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names

# 查看数据集基本信息
print("数据集前5行：")
print(df.head())
print(f"\n数据集总行数：{df.shape[0]}, 总列数：{df.shape[1]}")
print(f"\n标签类别分布：\n{df['class'].value_counts(normalize=True).round(4)*100}%")

# ------------------------------------------------------------------------------
# 2. 数据预处理 + 特征工程（新增组合特征，不增加维度爆炸）
# ------------------------------------------------------------------------------
# 定义每个特征的正确业务顺序，做有序编码
feature_order = {
    'buying':   ['low', 'med', 'high', 'vhigh'],
    'maint':    ['low', 'med', 'high', 'vhigh'],
    'doors':    ['2', '3', '4', '5more'],
    'persons':  ['2', '4', 'more'],
    'lug_boot': ['small', 'med', 'big'],
    'safety':   ['low', 'med', 'high']
}

# 对特征列做有序编码
oe = OrdinalEncoder(categories=[feature_order[col] for col in feature_order.keys()])
X_encoded = pd.DataFrame(oe.fit_transform(df[feature_order.keys()]), columns=feature_order.keys())

# 新增高价值组合特征（基于业务逻辑，不引入噪声）
# 总拥有成本：购买价+维修价
X_encoded['total_cost'] = X_encoded['buying'] + X_encoded['maint']
# 综合实用性：载客数+后备箱大小
X_encoded['utility'] = X_encoded['persons'] + X_encoded['lug_boot']
# 性价比：实用性 / 总拥有成本（加1避免除零）
X_encoded['cost_performance'] = X_encoded['utility'] / (X_encoded['total_cost'] + 1)

# 对标签列单独做标签编码
le = LabelEncoder()
y = le.fit_transform(df['class'])

# 打印标签映射关系
class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"\n标签类别映射关系：{class_mapping}")

# ------------------------------------------------------------------------------
# 3. 划分训练集与测试集（分层抽样保证分布一致）
# ------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------------------------------------------------------------------
# 4. 模型训练：Bagging集成多个深度为4的决策树（核心优化）
# ------------------------------------------------------------------------------
# 先找到单个决策树的最优参数（max_depth=4固定）
base_tree = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

# 网格搜索最优参数（不改变max_depth）
param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3],
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(
    estimator=base_tree,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_base_tree = grid_search.best_estimator_

print(f"\n单个决策树最优参数：{grid_search.best_params_}")
print(f"单个决策树交叉验证准确率：{grid_search.best_score_:.4f}")

# 使用Bagging集成多个最优决策树（每个树深度仍为4）
# n_estimators=50：50个弱模型投票，方差更低，准确率更高
bagging_model = BaggingClassifier(
    estimator=best_base_tree,
    n_estimators=50,
    max_samples=0.8,  # 每个树使用80%的训练样本
    max_features=0.8,  # 每个树使用80%的特征
    bootstrap=True,
    bootstrap_features=False,
    random_state=42,
    n_jobs=-1
)

# 拟合集成模型
bagging_model.fit(X_train, y_train)

# ------------------------------------------------------------------------------
# 5. 模型评估
# ------------------------------------------------------------------------------
# 单个最优决策树评估
y_train_pred_single = best_base_tree.predict(X_train)
train_accuracy_single = accuracy_score(y_train, y_train_pred_single)
y_test_pred_single = best_base_tree.predict(X_test)
test_accuracy_single = accuracy_score(y_test, y_test_pred_single)

# Bagging集成模型评估
y_train_pred_bagging = bagging_model.predict(X_train)
train_accuracy_bagging = accuracy_score(y_train, y_train_pred_bagging)
y_test_pred_bagging = bagging_model.predict(X_test)
test_accuracy_bagging = accuracy_score(y_test, y_test_pred_bagging)

# 打印核心评估指标对比
print("\n" + "="*80)
print("【单个最优决策树 vs Bagging集成模型 性能对比】")
print("="*80)
print(f"单个决策树 - 训练集准确率: {train_accuracy_single:.4f}")
print(f"单个决策树 - 测试集准确率: {test_accuracy_single:.4f}")
print(f"单个决策树 - 过拟合程度: {train_accuracy_single - test_accuracy_single:.4f}")
print("-"*80)
print(f"Bagging集成模型 - 训练集准确率: {train_accuracy_bagging:.4f}")
print(f"Bagging集成模型 - 测试集准确率: {test_accuracy_bagging:.4f}")
print(f"Bagging集成模型 - 过拟合程度: {train_accuracy_bagging - test_accuracy_bagging:.4f}")
print("="*80)

# 过拟合判断
if train_accuracy_bagging - test_accuracy_bagging > 0.1:
    print("⚠️  警告：模型存在明显过拟合")
else:
    print("✅  模型拟合状态良好，无明显过拟合")
print("="*80)

# 集成模型详细分类报告
print("Bagging集成模型测试集详细分类报告：")
print(classification_report(
    y_test, y_test_pred_bagging,
    target_names=le.classes_,
    zero_division=0
))
print("="*80)
print("Bagging集成模型测试集混淆矩阵：")
print(confusion_matrix(y_test, y_test_pred_bagging))

# ------------------------------------------------------------------------------
# 6. 可视化单个最优决策树（集成模型无法直接可视化，展示基础树）
# ------------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(30, 18), dpi=300)
plot_tree(
    best_base_tree,
    feature_names=X_encoded.columns,
    class_names=le.classes_,
    filled=True,
    rounded=True,
    fontsize=8
)
# 构建图片保存路径
image_path = os.path.join(script_dir, 'car_evaluation_best_single_tree.png')
plt.savefig(image_path, bbox_inches='tight')
plt.show()