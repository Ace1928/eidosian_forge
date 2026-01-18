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
@pytest.mark.parametrize('coo_container', COO_CONTAINERS)
def test_linear_regression_sparse_multiple_outcome(global_random_seed, coo_container):
    rng = np.random.RandomState(global_random_seed)
    X, y = make_sparse_uncorrelated(random_state=rng)
    X = coo_container(X)
    Y = np.vstack((y, y)).T
    n_features = X.shape[1]
    ols = LinearRegression()
    ols.fit(X, Y)
    assert ols.coef_.shape == (2, n_features)
    Y_pred = ols.predict(X)
    ols.fit(X, y.ravel())
    y_pred = ols.predict(X)
    assert_array_almost_equal(np.vstack((y_pred, y_pred)).T, Y_pred, decimal=3)