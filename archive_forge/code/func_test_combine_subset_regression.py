import os
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.sandbox.regression.penalized import TheilGLS
def test_combine_subset_regression(self):
    endog = self.endog
    exog = self.exog
    nobs = len(endog)
    n05 = nobs // 2
    np.random.seed(987125)
    shuffle_idx = np.random.permutation(np.arange(nobs))
    ys = endog[shuffle_idx]
    xs = exog[shuffle_idx]
    k = 10
    res_ols0 = OLS(ys[:n05], xs[:n05, :k]).fit()
    res_ols1 = OLS(ys[n05:], xs[n05:, :k]).fit()
    w = res_ols1.scale / res_ols0.scale
    mod_1 = TheilGLS(ys[n05:], xs[n05:, :k], r_matrix=np.eye(k), q_matrix=res_ols0.params, sigma_prior=w * res_ols0.cov_params())
    res_1p = mod_1.fit(cov_type='data-prior')
    res_1s = mod_1.fit(cov_type='sandwich')
    res_olsf = OLS(ys, xs[:, :k]).fit()
    assert_allclose(res_1p.params, res_olsf.params, rtol=1e-09)
    corr_fact = np.sqrt(res_1p.scale / res_olsf.scale)
    assert_allclose(res_1p.bse, res_olsf.bse * corr_fact, rtol=0.001)
    bse1 = np.array([0.26589869, 0.15224812, 0.38407399, 0.75679949, 0.660842, 0.5417408, 0.53697607, 0.66006377, 0.38228551, 0.53920485])
    assert_allclose(res_1s.bse, bse1, rtol=1e-07)