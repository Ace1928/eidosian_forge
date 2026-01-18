import pytest
import numpy as np
import numpy.testing as npt
import statsmodels.api as sm
@pytest.mark.slow
def test_censored_ll_cvls(self):
    nobs = 200
    np.random.seed(1234)
    C1 = np.random.normal(size=(nobs,))
    C2 = np.random.normal(2, 1, size=(nobs,))
    noise = np.random.normal(size=(nobs,))
    Y = 0.3 + 1.2 * C1 - 0.9 * C2 + noise
    Y[Y > 0] = 0
    model = nparam.KernelCensoredReg(endog=[Y], exog=[C1, C2], reg_type='ll', var_type='cc', bw='cv_ls', censor_val=0)
    sm_mean, sm_mfx = model.fit()
    npt.assert_allclose(sm_mfx[0, :], [1.2, -0.9], rtol=0.2)