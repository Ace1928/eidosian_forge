import pytest
import numpy as np
import numpy.testing as npt
import statsmodels.api as sm
def test_mixed_mfx_ll_cvls(self, file_name='RegData.csv'):
    nobs = 200
    np.random.seed(1234)
    ovals = np.random.binomial(2, 0.5, size=(nobs,))
    C1 = np.random.normal(size=(nobs,))
    C2 = np.random.normal(2, 1, size=(nobs,))
    noise = np.random.normal(size=(nobs,))
    b0 = 3
    b1 = 1.2
    b2 = 3.7
    b3 = 2.3
    Y = b0 + b1 * C1 + b2 * C2 + b3 * ovals + noise
    bw_cv_ls = np.array([1.04726, 1.67485, 0.39852])
    model = nparam.KernelReg(endog=[Y], exog=[C1, C2, ovals], reg_type='ll', var_type='cco', bw=bw_cv_ls)
    sm_mean, sm_mfx = model.fit()
    sm_R2 = model.r_squared()
    npt.assert_allclose(sm_mfx[0, :], [b1, b2, b3], rtol=0.2)