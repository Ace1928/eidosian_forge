from inspect import signature
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
@pytest.mark.parametrize('kernel', [kernel for kernel in kernels if kernel != kernel_rbf_plus_white])
def test_auto_vs_cross(kernel):
    K_auto = kernel(X)
    K_cross = kernel(X, X)
    assert_almost_equal(K_auto, K_cross, 5)