import pytest
import numpy as np
import numpy.testing as npt
import statsmodels.api as sm
def test_efficient_user_specificed_bw(self):
    bw_user = [0.23, 434697.22]
    model = nparam.KernelReg(endog=[self.y], exog=[self.c1, self.c2], reg_type='lc', var_type='cc', bw=bw_user, defaults=nparam.EstimatorSettings(efficient=True))
    npt.assert_equal(model.bw, bw_user)