import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import statsmodels.api as sm
from statsmodels.tools import numdiff
from statsmodels.tools.numdiff import (
def test_grad_fun1_fd(self):
    for test_params in self.params:
        gtrue = self.gradtrue(test_params)
        fun = self.fun()
        epsilon = 1e-06
        gfd = numdiff.approx_fprime(test_params, fun, epsilon=epsilon, args=self.args)
        gfd += numdiff.approx_fprime(test_params, fun, epsilon=-epsilon, args=self.args)
        gfd /= 2.0
        assert_almost_equal(gtrue, gfd, decimal=DEC6)