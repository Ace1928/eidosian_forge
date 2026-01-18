import warnings
import numpy as np
import pytest
from scipy.optimize import approx_fprime
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel
from sklearn.utils._testing import assert_almost_equal, assert_array_equal
@pytest.mark.parametrize('kernel', kernels)
def test_lml_gradient(kernel):
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    lml, lml_gradient = gpc.log_marginal_likelihood(kernel.theta, True)
    lml_gradient_approx = approx_fprime(kernel.theta, lambda theta: gpc.log_marginal_likelihood(theta, False), 1e-10)
    assert_almost_equal(lml_gradient, lml_gradient_approx, 3)