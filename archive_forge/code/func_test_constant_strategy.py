import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def test_constant_strategy():
    X = [[0], [0], [0], [0]]
    y = [2, 1, 2, 2]
    clf = DummyClassifier(strategy='constant', random_state=0, constant=1)
    clf.fit(X, y)
    assert_array_equal(clf.predict(X), np.ones(len(X)))
    _check_predict_proba(clf, X, y)
    X = [[0], [0], [0], [0]]
    y = ['two', 'one', 'two', 'two']
    clf = DummyClassifier(strategy='constant', random_state=0, constant='one')
    clf.fit(X, y)
    assert_array_equal(clf.predict(X), np.array(['one'] * 4))
    _check_predict_proba(clf, X, y)