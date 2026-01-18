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
def test_linear_regression_positive_vs_nonpositive_when_positive(global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    n_samples = 200
    n_features = 4
    X = rng.rand(n_samples, n_features)
    y = X[:, 0] + 2 * X[:, 1] + 3 * X[:, 2] + 1.5 * X[:, 3]
    reg = LinearRegression(positive=True)
    reg.fit(X, y)
    regn = LinearRegression(positive=False)
    regn.fit(X, y)
    assert np.mean((reg.coef_ - regn.coef_) ** 2) < 1e-06