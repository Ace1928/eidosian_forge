import warnings
import numpy as np
import pytest
from scipy import linalg, sparse
from sklearn.datasets import load_iris, make_regression, make_sparse_uncorrelated
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import (
from sklearn.preprocessing import add_dummy_feature
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
def test_linear_regression_multiple_outcome():
    rng = np.random.RandomState(0)
    X, y = make_regression(random_state=rng)
    Y = np.vstack((y, y)).T
    n_features = X.shape[1]
    reg = LinearRegression()
    reg.fit(X, Y)
    assert reg.coef_.shape == (2, n_features)
    Y_pred = reg.predict(X)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    assert_array_almost_equal(np.vstack((y_pred, y_pred)).T, Y_pred, decimal=3)