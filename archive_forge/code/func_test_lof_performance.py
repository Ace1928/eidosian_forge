import re
from math import sqrt
import numpy as np
import pytest
from sklearn import metrics, neighbors
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_lof_performance(global_dtype):
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2).astype(global_dtype, copy=False)
    X_train = X[:100]
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2)).astype(global_dtype, copy=False)
    X_test = np.r_[X[100:], X_outliers]
    y_test = np.array([0] * 20 + [1] * 20)
    clf = neighbors.LocalOutlierFactor(novelty=True).fit(X_train)
    y_pred = -clf.decision_function(X_test)
    assert roc_auc_score(y_test, y_pred) > 0.99