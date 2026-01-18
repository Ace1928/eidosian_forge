import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils._testing import assert_array_almost_equal, ignore_warnings
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_kernel_ridge_precomputed():
    for kernel in ['linear', 'rbf', 'poly', 'cosine']:
        K = pairwise_kernels(X, X, metric=kernel)
        pred = KernelRidge(kernel=kernel).fit(X, y).predict(X)
        pred2 = KernelRidge(kernel='precomputed').fit(K, y).predict(K)
        assert_array_almost_equal(pred, pred2)