import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils._testing import assert_array_almost_equal, ignore_warnings
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_kernel_ridge_singular_kernel():
    pred = Ridge(alpha=0, fit_intercept=False).fit(X, y).predict(X)
    kr = KernelRidge(kernel='linear', alpha=0)
    ignore_warnings(kr.fit)(X, y)
    pred2 = kr.predict(X)
    assert_array_almost_equal(pred, pred2)