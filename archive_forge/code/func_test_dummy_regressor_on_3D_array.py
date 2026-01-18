import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def test_dummy_regressor_on_3D_array():
    X = np.array([[['foo']], [['bar']], [['baz']]])
    y = np.array([2, 2, 2])
    y_expected = np.array([2, 2, 2])
    cls = DummyRegressor()
    cls.fit(X, y)
    y_pred = cls.predict(X)
    assert_array_equal(y_pred, y_expected)