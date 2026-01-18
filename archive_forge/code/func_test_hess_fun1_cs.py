import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import statsmodels.api as sm
from statsmodels.tools import numdiff
from statsmodels.tools.numdiff import (
def test_hess_fun1_cs(self):
    for test_params in self.params:
        hetrue = self.hesstrue(test_params)
        if hetrue is not None:
            fun = self.fun()
            hecs = numdiff.approx_hess_cs(test_params, fun, args=self.args)
            assert_almost_equal(hetrue, hecs, decimal=DEC6)