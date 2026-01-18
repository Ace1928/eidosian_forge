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
def test_custom_optimizer(kernel, global_random_seed):

    def optimizer(obj_func, initial_theta, bounds):
        rng = np.random.RandomState(global_random_seed)
        theta_opt, func_min = (initial_theta, obj_func(initial_theta, eval_gradient=False))
        for _ in range(10):
            theta = np.atleast_1d(rng.uniform(np.maximum(-2, bounds[:, 0]), np.minimum(1, bounds[:, 1])))
            f = obj_func(theta, eval_gradient=False)
            if f < func_min:
                theta_opt, func_min = (theta, f)
        return (theta_opt, func_min)
    gpc = GaussianProcessClassifier(kernel=kernel, optimizer=optimizer)
    gpc.fit(X, y_mc)
    assert gpc.log_marginal_likelihood(gpc.kernel_.theta) >= gpc.log_marginal_likelihood(kernel.theta)