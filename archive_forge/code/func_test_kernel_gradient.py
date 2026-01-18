from inspect import signature
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
@pytest.mark.parametrize('kernel', kernels)
def test_kernel_gradient(kernel):
    K, K_gradient = kernel(X, eval_gradient=True)
    assert K_gradient.shape[0] == X.shape[0]
    assert K_gradient.shape[1] == X.shape[0]
    assert K_gradient.shape[2] == kernel.theta.shape[0]

    def eval_kernel_for_theta(theta):
        kernel_clone = kernel.clone_with_theta(theta)
        K = kernel_clone(X, eval_gradient=False)
        return K
    K_gradient_approx = _approx_fprime(kernel.theta, eval_kernel_for_theta, 1e-10)
    assert_almost_equal(K_gradient, K_gradient_approx, 4)