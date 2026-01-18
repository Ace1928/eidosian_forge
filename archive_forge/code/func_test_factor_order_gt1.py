from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo
def test_factor_order_gt1():
    index_M = pd.period_range(start='2000', periods=12, freq='M')
    index_Q = pd.period_range(start='2000', periods=4, freq='Q')
    dta_M = pd.DataFrame(np.zeros((12, 2)), index=index_M, columns=['M0', 'M1'])
    dta_Q = pd.DataFrame(np.zeros((4, 2)), index=index_Q, columns=['Q0', 'Q1'])
    dta_M.iloc[0] = 1.0
    dta_Q.iloc[1] = 1.0
    mod = dynamic_factor_mq.DynamicFactorMQ(dta_M, endog_quarterly=dta_Q, factors=1, factor_orders=6, idiosyncratic_ar1=True)
    assert_equal(mod.k_endog, 2 + 2)
    assert_equal(mod.k_states, 6 + 2 + 2 * 5)
    assert_equal(mod.ssm.k_posdef, 1 + 2 + 2)
    assert_equal(mod.endog_names, ['M0', 'M1', 'Q0', 'Q1'])
    desired = ['0', 'L1.0', 'L2.0', 'L3.0', 'L4.0', 'L5.0'] + ['eps_M.M0', 'eps_M.M1', 'eps_Q.Q0', 'eps_Q.Q1'] + ['L1.eps_Q.Q0', 'L1.eps_Q.Q1'] + ['L2.eps_Q.Q0', 'L2.eps_Q.Q1'] + ['L3.eps_Q.Q0', 'L3.eps_Q.Q1'] + ['L4.eps_Q.Q0', 'L4.eps_Q.Q1']
    assert_equal(mod.state_names, desired)
    desired = ['loading.0->M0', 'loading.0->M1', 'loading.0->Q0', 'loading.0->Q1', 'L1.0->0', 'L2.0->0', 'L3.0->0', 'L4.0->0', 'L5.0->0', 'L6.0->0', 'fb(0).cov.chol[1,1]', 'L1.eps_M.M0', 'L1.eps_M.M1', 'L1.eps_Q.Q0', 'L1.eps_Q.Q1', 'sigma2.M0', 'sigma2.M1', 'sigma2.Q0', 'sigma2.Q1']
    assert_equal(mod.param_names, desired)
    assert_allclose(mod['obs_intercept'], 0)
    assert_allclose(mod['design', :2, 6:8], np.eye(2))
    assert_allclose(mod['design', 2:, 8:10], np.eye(2))
    assert_allclose(mod['design', 2:, 10:12], 2 * np.eye(2))
    assert_allclose(mod['design', 2:, 12:14], 3 * np.eye(2))
    assert_allclose(mod['design', 2:, 14:16], 2 * np.eye(2))
    assert_allclose(mod['design', 2:, 16:18], np.eye(2))
    assert_allclose(np.sum(mod['design']), 20)
    assert_allclose(mod['obs_cov'], 0)
    assert_allclose(mod['state_intercept'], 0)
    assert_allclose(mod['transition', 1:6, :5], np.eye(5))
    assert_allclose(mod['transition', 10:18, 8:16], np.eye(2 * 4))
    assert_allclose(np.sum(mod['transition']), 13)
    assert_allclose(mod['selection', 0, 0], np.eye(1))
    assert_allclose(mod['selection', 6:8, 1:3], np.eye(2))
    assert_allclose(mod['selection', 8:10, 3:5], np.eye(2))
    assert_allclose(np.sum(mod['selection']), 5)
    assert_allclose(mod['state_cov'], 0)
    mod.update(np.arange(mod.k_params) + 2)
    assert_allclose(mod['obs_intercept'], 0)
    desired = np.array([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0, 0.0, 0.0], [4.0, 8.0, 12, 8.0, 4.0, 0.0], [5.0, 10, 15, 10, 5.0, 0.0]])
    assert_allclose(mod['design', :, :6], desired)
    desired = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0]])
    assert_allclose(mod['design', :, 6:], desired)
    assert_allclose(mod['obs_cov'], 0)
    assert_allclose(mod['state_intercept'], 0)
    desired = np.array([[6.0, 7.0, 8.0, 9.0, 10, 11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 0]])
    assert_allclose(mod['transition'], desired)
    assert_allclose(np.sum(mod['selection']), 5)
    desired = np.array([[144, 0.0, 0.0, 0.0, 0.0], [0.0, 17.0, 0.0, 0.0, 0.0], [0.0, 0.0, 18.0, 0.0, 0.0], [0.0, 0.0, 0.0, 19.0, 0.0], [0.0, 0.0, 0.0, 0.0, 20.0]])
    assert_allclose(mod['state_cov'], desired)