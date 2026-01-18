import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils._testing import assert_array_almost_equal, ignore_warnings
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_kernel_ridge_sample_weights():
    K = np.dot(X, X.T)
    sw = np.random.RandomState(0).rand(X.shape[0])
    pred = Ridge(alpha=1, fit_intercept=False).fit(X, y, sample_weight=sw).predict(X)
    pred2 = KernelRidge(kernel='linear', alpha=1).fit(X, y, sample_weight=sw).predict(X)
    pred3 = KernelRidge(kernel='precomputed', alpha=1).fit(K, y, sample_weight=sw).predict(K)
    assert_array_almost_equal(pred, pred2)
    assert_array_almost_equal(pred, pred3)