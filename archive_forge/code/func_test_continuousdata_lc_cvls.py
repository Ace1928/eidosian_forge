import pytest
import numpy as np
import numpy.testing as npt
import statsmodels.api as sm
def test_continuousdata_lc_cvls(self):
    model = nparam.KernelReg(endog=[self.y], exog=[self.c1, self.c2], reg_type='lc', var_type='cc', bw='cv_ls')
    sm_bw = model.bw
    R_bw = [0.6163835, 0.1649656]
    sm_mean, sm_mfx = model.fit()
    sm_mean = sm_mean[0:5]
    sm_mfx = sm_mfx[0:5]
    R_mean = [31.49157, 37.29536, 43.72332, 40.58997, 36.80711]
    sm_R2 = model.r_squared()
    R_R2 = 0.956381720885
    npt.assert_allclose(sm_bw, R_bw, atol=0.01)
    npt.assert_allclose(sm_mean, R_mean, atol=0.01)
    npt.assert_allclose(sm_R2, R_R2, atol=0.01)