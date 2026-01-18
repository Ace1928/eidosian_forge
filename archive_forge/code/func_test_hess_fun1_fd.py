import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import statsmodels.api as sm
from statsmodels.tools import numdiff
from statsmodels.tools.numdiff import (
def test_hess_fun1_fd(self):
    for test_params in self.params:
        hetrue = self.hesstrue(test_params)
        if hetrue is not None:
            fun = self.fun()
            hefd = numdiff.approx_hess1(test_params, fun, args=self.args)
            assert_almost_equal(hetrue, hefd, decimal=DEC3)
            hefd = numdiff.approx_hess2(test_params, fun, args=self.args)
            assert_almost_equal(hetrue, hefd, decimal=DEC3)
            hefd = numdiff.approx_hess3(test_params, fun, args=self.args)
            assert_almost_equal(hetrue, hefd, decimal=DEC3)