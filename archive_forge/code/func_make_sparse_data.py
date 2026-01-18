import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, LIL_CONTAINERS
def make_sparse_data(sparse_container, n_samples=100, n_features=100, n_informative=10, seed=42, positive=False, n_targets=1):
    random_state = np.random.RandomState(seed)
    w = random_state.randn(n_features, n_targets)
    w[n_informative:] = 0.0
    if positive:
        w = np.abs(w)
    X = random_state.randn(n_samples, n_features)
    rnd = random_state.uniform(size=(n_samples, n_features))
    X[rnd > 0.5] = 0.0
    y = np.dot(X, w)
    X = sparse_container(X)
    if n_targets == 1:
        y = np.ravel(y)
    return (X, y)