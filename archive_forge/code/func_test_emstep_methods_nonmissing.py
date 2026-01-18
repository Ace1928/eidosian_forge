from statsmodels.compat.pandas import QUARTER_END
import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.io import matlab
from statsmodels.tsa.statespace import dynamic_factor_mq, initialization
@pytest.mark.parametrize('k_factors,factor_orders,factor_multiplicities,idiosyncratic_ar1', [(1, 1, 1, True), (3, 1, 1, True), (1, 6, 1, True), (3, {('0', '1', '2'): 6}, 1, True), (1, 1, 1, False), (3, 1, 1, False), (1, 6, 1, False), (3, {('0', '1', '2'): 6}, 1, False), (BLOCK_FACTORS_KP1.copy(), BLOCK_FACTOR_ORDERS_KP1.copy(), 1, True), (BLOCK_FACTORS_KP1.copy(), BLOCK_FACTOR_ORDERS_KP1.copy(), 1, False), (BLOCK_FACTORS_KP1.copy(), BLOCK_FACTOR_ORDERS_KP2.copy(), BLOCK_FACTOR_MULTIPLICITIES_KP2, True), (BLOCK_FACTORS_KP1.copy(), BLOCK_FACTOR_ORDERS_KP2.copy(), BLOCK_FACTOR_MULTIPLICITIES_KP2, False)])
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_emstep_methods_nonmissing(matlab_results, k_factors, factor_orders, factor_multiplicities, idiosyncratic_ar1):
    dta_M = matlab_results[0].iloc[:, :8]
    dta_M = (dta_M - dta_M.mean()) / dta_M.std()
    endog_M = dta_M.interpolate().bfill()
    if isinstance(k_factors, dict):
        if 'GDPC1' in k_factors:
            del k_factors['GDPC1']
        if 'ULCNFB' in k_factors:
            del k_factors['ULCNFB']
    mod = dynamic_factor_mq.DynamicFactorMQ(endog_M, factors=k_factors, factor_orders=factor_orders, factor_multiplicities=factor_multiplicities, idiosyncratic_ar1=idiosyncratic_ar1)
    mod.ssm.filter_univariate = True
    params0 = mod.start_params
    _, params1 = mod._em_iteration(params0, mstep_method='missing')
    _, params1_nonmissing = mod._em_iteration(params0, mstep_method='nonmissing')
    assert_allclose(params1_nonmissing, params1, atol=1e-13)
    mod.update(params1)
    res = mod.ssm.smooth()
    a = res.smoothed_state.T[..., None]
    cov_a = res.smoothed_state_cov.transpose(2, 0, 1)
    Eaa = cov_a + np.matmul(a, a.transpose(0, 2, 1))
    Lambda, H = mod._em_maximization_obs_missing(res, Eaa, a, compute_H=True)
    Lambda_nonmissing, H_nonmissing = mod._em_maximization_obs_nonmissing(res, Eaa, a, compute_H=True)
    assert_allclose(Lambda_nonmissing, Lambda, atol=1e-13)
    assert_allclose(H_nonmissing, H, atol=1e-13)