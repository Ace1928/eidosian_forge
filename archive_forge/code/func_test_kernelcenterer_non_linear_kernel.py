import re
import warnings
import numpy as np
import numpy.linalg as la
import pytest
from scipy import sparse, stats
from sklearn import datasets
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale
from sklearn.svm import SVR
from sklearn.utils import gen_batches, shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import (
from sklearn.utils.sparsefuncs import mean_variance_axis
def test_kernelcenterer_non_linear_kernel():
    """Check kernel centering for non-linear kernel."""
    rng = np.random.RandomState(0)
    X, X_test = (rng.randn(100, 50), rng.randn(20, 50))

    def phi(X):
        """Our mapping function phi."""
        return np.vstack([np.clip(X, a_min=0, a_max=None), -np.clip(X, a_min=None, a_max=0)])
    phi_X = phi(X)
    phi_X_test = phi(X_test)
    scaler = StandardScaler(with_std=False)
    phi_X_center = scaler.fit_transform(phi_X)
    phi_X_test_center = scaler.transform(phi_X_test)
    K = phi_X @ phi_X.T
    K_test = phi_X_test @ phi_X.T
    K_center = phi_X_center @ phi_X_center.T
    K_test_center = phi_X_test_center @ phi_X_center.T
    kernel_centerer = KernelCenterer()
    kernel_centerer.fit(K)
    assert_allclose(kernel_centerer.transform(K), K_center)
    assert_allclose(kernel_centerer.transform(K_test), K_test_center)
    ones_M = np.ones_like(K) / K.shape[0]
    K_centered = K - ones_M @ K - K @ ones_M + ones_M @ K @ ones_M
    assert_allclose(kernel_centerer.transform(K), K_centered)
    ones_prime_M = np.ones_like(K_test) / K.shape[0]
    K_test_centered = K_test - ones_prime_M @ K - K_test @ ones_M + ones_prime_M @ K @ ones_M
    assert_allclose(kernel_centerer.transform(K_test), K_test_centered)