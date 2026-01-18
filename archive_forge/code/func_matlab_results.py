from statsmodels.compat.pandas import QUARTER_END
import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.io import matlab
from statsmodels.tsa.statespace import dynamic_factor_mq, initialization
@pytest.fixture(scope='module')
def matlab_results():
    results = {}
    for run in ['111', '112', '11F', '221', '222', '22F']:
        res = matlab.loadmat(os.path.join(results_path, f'test_dfm_{run}.mat'))
        k_factors = res['Spec']['k'][0, 0][0, 0]
        factor_orders = res['Spec']['p'][0, 0][0, 0]
        _factor_orders = max(5, factor_orders)
        idio = k_factors * _factor_orders + 3
        ix = np.r_[np.arange(idio), idio + np.arange(10).reshape(2, 5).ravel(order='F')]
        initial_state = res['Res']['Z_0'][0, 0][:, 0][ix]
        initial_state_cov = res['Res']['V_0'][0, 0][ix][:, ix]
        if k_factors == 2:
            factor_orders = {('0', '1'): factor_orders}
        results[run] = {'k_endog_M': 3, 'factors': k_factors, 'factor_orders': factor_orders, 'factor_multiplicities': None, 'params': res['params'][:, 0], 'llf': res['Res']['loglik'][0, 0][0, 0], 'initial_state': initial_state, 'initial_state_cov': initial_state_cov, 'smoothed_forecasts': res['Res']['x_sm'][0, 0]}
    for run in ['112', '222']:
        res = matlab.loadmat(os.path.join(results_path, f'test_news_{run}.mat'))
        k_factors = res['Spec']['k'][0, 0][0, 0]
        factor_orders = res['Spec']['p'][0, 0][0, 0]
        _factor_orders = max(5, factor_orders)
        idio = k_factors * _factor_orders + 3
        ix = np.r_[np.arange(idio), idio + np.arange(10).reshape(2, 5).ravel(order='F')]
        initial_state = res['Res']['Z_0'][0, 0][:, 0][ix]
        initial_state_cov = res['Res']['V_0'][0, 0][ix][:, ix]
        if k_factors == 2:
            factor_orders = {('0', '1'): factor_orders}
        results[f'news_{run}'] = {'k_endog_M': 3, 'factors': k_factors, 'factor_orders': factor_orders, 'factor_multiplicities': None, 'params': res['params'][:, 0], 'initial_state': initial_state, 'initial_state_cov': initial_state_cov, 'revision_impacts': res['Res']['impact_revisions'][0, 0], 'weight': res['Res']['weight'], 'news_table': res['Res']['news_table'][0, 0]}
    for run in ['111', '112', '221', '222']:
        res = matlab.loadmat(os.path.join(results_path, f'test_dfm_blocks_{run}.mat'))
        k_factors = res['Spec']['k'][0, 0][0, 0]
        factor_order = res['Spec']['p'][0, 0][0, 0]
        _factor_order = max(5, factor_order)
        idio = 3 * k_factors * _factor_order + 6
        ix = np.r_[np.arange(idio), idio + np.arange(10).reshape(2, 5).ravel(order='F')]
        initial_state = res['Res']['Z_0'][0, 0][:, 0][ix]
        initial_state_cov = res['Res']['V_0'][0, 0][ix][:, ix]
        if k_factors == 1:
            factors = BLOCK_FACTORS_KP1.copy()
            factor_orders = BLOCK_FACTOR_ORDERS_KP1.copy()
            factor_multiplicities = None
        else:
            factors = BLOCK_FACTORS_KP1.copy()
            factor_orders = BLOCK_FACTOR_ORDERS_KP2.copy()
            factor_multiplicities = BLOCK_FACTOR_MULTIPLICITIES_KP2.copy()
        results[f'block_{run}'] = {'k_endog_M': 6, 'factors': factors, 'factor_orders': factor_orders, 'factor_multiplicities': factor_multiplicities, 'params': res['params'][:, 0], 'llf': res['Res']['loglik'][0, 0][0, 0], 'initial_state': initial_state, 'initial_state_cov': initial_state_cov, 'smoothed_forecasts': res['Res']['x_sm'][0, 0]}
    for run in ['112', '222']:
        res = matlab.loadmat(os.path.join(results_path, f'test_news_blocks_{run}.mat'))
        k_factors = res['Spec']['k'][0, 0][0, 0]
        factor_order = res['Spec']['p'][0, 0][0, 0]
        _factor_order = max(5, factor_order)
        idio = 3 * k_factors * _factor_order + 6
        ix = np.r_[np.arange(idio), idio + np.arange(10).reshape(2, 5).ravel(order='F')]
        initial_state = res['Res']['Z_0'][0, 0][:, 0][ix]
        initial_state_cov = res['Res']['V_0'][0, 0][ix][:, ix]
        if k_factors == 1:
            factors = BLOCK_FACTORS_KP1.copy()
            factor_orders = BLOCK_FACTOR_ORDERS_KP1.copy()
            factor_multiplicities = None
        else:
            factors = BLOCK_FACTORS_KP1.copy()
            factor_orders = BLOCK_FACTOR_ORDERS_KP2.copy()
            factor_multiplicities = BLOCK_FACTOR_MULTIPLICITIES_KP2.copy()
        results[f'news_block_{run}'] = {'k_endog_M': 6, 'factors': factors, 'factor_orders': factor_orders, 'factor_multiplicities': factor_multiplicities, 'params': res['params'][:, 0], 'initial_state': initial_state, 'initial_state_cov': initial_state_cov, 'revision_impacts': res['Res']['impact_revisions'][0, 0], 'weight': res['Res']['weight'], 'news_table': res['Res']['news_table'][0, 0]}

    def get_data(us_data, mean_M=None, std_M=None, mean_Q=None, std_Q=None):
        dta_M = us_data[['CPIAUCSL', 'UNRATE', 'PAYEMS', 'RSAFS', 'TTLCONS', 'TCU']].copy()
        dta_Q = us_data[['GDPC1', 'ULCNFB']].copy()
        dta_Q.index = dta_Q.index.to_timestamp()
        dta_Q = dta_Q.resample(QUARTER_END).last()
        dta_Q.index = dta_Q.index.to_period()
        dta_M['CPIAUCSL'] = (dta_M['CPIAUCSL'] / dta_M['CPIAUCSL'].shift(1) - 1) * 100
        dta_M['UNRATE'] = dta_M['UNRATE'].diff()
        dta_M['PAYEMS'] = dta_M['PAYEMS'].diff()
        dta_M['TCU'] = dta_M['TCU'].diff()
        dta_M['RSAFS'] = (dta_M['RSAFS'] / dta_M['RSAFS'].shift(1) - 1) * 100
        dta_M['TTLCONS'] = (dta_M['TTLCONS'] / dta_M['TTLCONS'].shift(1) - 1) * 100
        dta_Q = ((dta_Q / dta_Q.shift(1)) ** 4 - 1) * 100
        start = '2000'
        dta_M = dta_M.loc[start:]
        dta_Q = dta_Q.loc[start:]
        first_ix = dta_M.first_valid_index()
        last_ix = dta_M.last_valid_index()
        dta_M = dta_M.loc[first_ix:last_ix]
        first_ix = dta_Q.first_valid_index()
        last_ix = dta_Q.last_valid_index()
        dta_Q = dta_Q.loc[first_ix:last_ix]
        return (dta_M, dta_Q)
    endog_M, endog_Q = get_data(us_data)
    updated_M, updated_Q = get_data(us_data_update)
    return (endog_M, endog_Q, results, updated_M, updated_Q)