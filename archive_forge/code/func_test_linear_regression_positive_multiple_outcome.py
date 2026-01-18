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
def test_linear_regression_positive_multiple_outcome(global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    X, y = make_sparse_uncorrelated(random_state=rng)
    Y = np.vstack((y, y)).T
    n_features = X.shape[1]
    ols = LinearRegression(positive=True)
    ols.fit(X, Y)
    assert ols.coef_.shape == (2, n_features)
    assert np.all(ols.coef_ >= 0.0)
    Y_pred = ols.predict(X)
    ols.fit(X, y.ravel())
    y_pred = ols.predict(X)
    assert_allclose(np.vstack((y_pred, y_pred)).T, Y_pred)