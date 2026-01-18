from inspect import signature
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
@pytest.mark.parametrize('kernel', kernels)
def test_kernel_versus_pairwise(kernel):
    if kernel != kernel_rbf_plus_white:
        K1 = kernel(X)
        K2 = pairwise_kernels(X, metric=kernel)
        assert_array_almost_equal(K1, K2)
    K1 = kernel(X, Y)
    K2 = pairwise_kernels(X, Y, metric=kernel)
    assert_array_almost_equal(K1, K2)