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
def test_linear_regression_sparse(global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    n = 100
    X = sparse.eye(n, n)
    beta = rng.rand(n)
    y = X @ beta
    ols = LinearRegression()
    ols.fit(X, y.ravel())
    assert_array_almost_equal(beta, ols.coef_ + ols.intercept_)
    assert_array_almost_equal(ols.predict(X) - y.ravel(), 0)