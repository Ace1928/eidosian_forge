import pytest
import numpy as np
import numpy.testing as npt
import statsmodels.api as sm
def test_censored_user_specified_kernel(self):
    model = nparam.KernelCensoredReg(endog=[self.y], exog=[self.c1, self.c2], reg_type='ll', var_type='cc', bw='cv_ls', censor_val=0, ckertype='tricube')
    sm_bw = model.bw
    R_bw = [0.581663, 0.5652]
    sm_mean, sm_mfx = model.fit()
    sm_mean = sm_mean[0:5]
    sm_mfx = sm_mfx[0:5]
    R_mean = [29.205526, 29.538008, 31.667581, 31.978866, 30.926714]
    sm_R2 = model.r_squared()
    R_R2 = 0.934825
    npt.assert_allclose(sm_bw, R_bw, atol=0.01)
    npt.assert_allclose(sm_mean, R_mean, atol=0.01)
    npt.assert_allclose(sm_R2, R_R2, atol=0.01)