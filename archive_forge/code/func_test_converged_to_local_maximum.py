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
@pytest.mark.parametrize('kernel', non_fixed_kernels)
def test_converged_to_local_maximum(kernel):
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    lml, lml_gradient = gpc.log_marginal_likelihood(gpc.kernel_.theta, True)
    assert np.all((np.abs(lml_gradient) < 0.0001) | (gpc.kernel_.theta == gpc.kernel_.bounds[:, 0]) | (gpc.kernel_.theta == gpc.kernel_.bounds[:, 1]))