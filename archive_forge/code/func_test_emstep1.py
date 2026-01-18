from statsmodels.compat.pandas import QUARTER_END
import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.io import matlab
from statsmodels.tsa.statespace import dynamic_factor_mq, initialization
@pytest.mark.parametrize('run', ['11', '22', 'block_11', 'block_22'])
def test_emstep1(matlab_results, run):
    endog_M, endog_Q = matlab_results[:2]
    results1 = matlab_results[2][f'{run}1']
    results2 = matlab_results[2][f'{run}2']
    mod = dynamic_factor_mq.DynamicFactorMQ(endog_M.iloc[:, :results1['k_endog_M']], endog_quarterly=endog_Q, factors=results1['factors'], factor_orders=results1['factor_orders'], factor_multiplicities=results1['factor_multiplicities'], idiosyncratic_ar1=True, init_t0=True, obs_cov_diag=True, standardize=True)
    init = initialization.Initialization(mod.k_states, 'known', constant=results1['initial_state'], stationary_cov=results1['initial_state_cov'])
    res2, params2 = mod._em_iteration(results1['params'], init=init, mstep_method='missing')
    true2 = results2['params']
    assert_allclose(params2[mod._p['loadings']], true2[mod._p['loadings']])
    assert_allclose(params2[mod._p['factor_ar']], true2[mod._p['factor_ar']])
    assert_allclose(params2[mod._p['factor_cov']], true2[mod._p['factor_cov']])
    assert_allclose(params2[mod._p['idiosyncratic_ar1']], true2[mod._p['idiosyncratic_ar1']])
    assert_allclose(params2[mod._p['idiosyncratic_var']], true2[mod._p['idiosyncratic_var']])