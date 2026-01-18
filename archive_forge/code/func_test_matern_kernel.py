from inspect import signature
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
def test_matern_kernel():
    K = Matern(nu=1.5, length_scale=1.0)(X)
    assert_array_almost_equal(np.diag(K), np.ones(X.shape[0]))
    K_absexp = np.exp(-euclidean_distances(X, X, squared=False))
    K = Matern(nu=0.5, length_scale=1.0)(X)
    assert_array_almost_equal(K, K_absexp)
    K_rbf = RBF(length_scale=1.0)(X)
    K = Matern(nu=np.inf, length_scale=1.0)(X)
    assert_array_almost_equal(K, K_rbf)
    assert_allclose(K, K_rbf)
    tiny = 1e-10
    for nu in [0.5, 1.5, 2.5]:
        K1 = Matern(nu=nu, length_scale=1.0)(X)
        K2 = Matern(nu=nu + tiny, length_scale=1.0)(X)
        assert_array_almost_equal(K1, K2)
    large = 100
    K1 = Matern(nu=large, length_scale=1.0)(X)
    K2 = RBF(length_scale=1.0)(X)
    assert_array_almost_equal(K1, K2, decimal=2)