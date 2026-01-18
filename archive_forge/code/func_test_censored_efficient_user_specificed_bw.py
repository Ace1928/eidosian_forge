import pytest
import numpy as np
import numpy.testing as npt
import statsmodels.api as sm
def test_censored_efficient_user_specificed_bw(self):
    nobs = 200
    np.random.seed(1234)
    C1 = np.random.normal(size=(nobs,))
    C2 = np.random.normal(2, 1, size=(nobs,))
    noise = np.random.normal(size=(nobs,))
    Y = 0.3 + 1.2 * C1 - 0.9 * C2 + noise
    Y[Y > 0] = 0
    bw_user = [0.23, 434697.22]
    model = nparam.KernelCensoredReg(endog=[Y], exog=[C1, C2], reg_type='ll', var_type='cc', bw=bw_user, censor_val=0, defaults=nparam.EstimatorSettings(efficient=True))
    npt.assert_equal(model.bw, bw_user)