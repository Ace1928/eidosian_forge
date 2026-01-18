import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def test_constant_strategy_multioutput():
    X = [[0], [0], [0], [0]]
    y = np.array([[2, 3], [1, 3], [2, 3], [2, 0]])
    n_samples = len(X)
    clf = DummyClassifier(strategy='constant', random_state=0, constant=[1, 0])
    clf.fit(X, y)
    assert_array_equal(clf.predict(X), np.hstack([np.ones((n_samples, 1)), np.zeros((n_samples, 1))]))
    _check_predict_proba(clf, X, y)