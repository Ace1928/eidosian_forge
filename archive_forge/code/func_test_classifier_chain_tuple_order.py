import re
import numpy as np
import pytest
from joblib import cpu_count
from sklearn import datasets
from sklearn.base import ClassifierMixin, clone
from sklearn.datasets import (
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import jaccard_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import (
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('order_type', [list, np.array, tuple])
def test_classifier_chain_tuple_order(order_type):
    X = [[1, 2, 3], [4, 5, 6], [1.5, 2.5, 3.5]]
    y = [[3, 2], [2, 3], [3, 2]]
    order = order_type([1, 0])
    chain = ClassifierChain(RandomForestClassifier(), order=order)
    chain.fit(X, y)
    X_test = [[1.5, 2.5, 3.5]]
    y_test = [[3, 2]]
    assert_array_almost_equal(chain.predict(X_test), y_test)