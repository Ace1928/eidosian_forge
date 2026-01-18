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
def test_lml_without_cloning_kernel(kernel):
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    input_theta = np.ones(gpc.kernel_.theta.shape, dtype=np.float64)
    gpc.log_marginal_likelihood(input_theta, clone_kernel=False)
    assert_almost_equal(gpc.kernel_.theta, input_theta, 7)