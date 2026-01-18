import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import statsmodels.api as sm
from statsmodels.tools import numdiff
from statsmodels.tools.numdiff import (
def test_grad_fun1_fdc(self):
    for test_params in self.params:
        gtrue = self.gradtrue(test_params)
        fun = self.fun()
        gfd = numdiff.approx_fprime(test_params, fun, epsilon=1e-08, args=self.args, centered=True)
        assert_almost_equal(gtrue, gfd, decimal=DEC5)