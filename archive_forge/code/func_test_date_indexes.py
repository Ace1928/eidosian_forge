from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo
@pytest.mark.parametrize('freq_Q', [QUARTER_END, f'{QUARTER_END}-DEC', f'{QUARTER_END}-JAN', 'QS', 'QS-DEC', 'QS-APR'])
@pytest.mark.parametrize('freq_M', [MONTH_END, 'MS'])
def test_date_indexes(reset_randomstate, freq_M, freq_Q):
    nobs_M = 10
    dates_M = pd.date_range(start='2000', periods=nobs_M, freq=freq_M)
    periods_M = pd.period_range(start='2000', periods=nobs_M, freq='M')
    dta_M = np.random.normal(size=(nobs_M, 2))
    endog_period_M = pd.DataFrame(dta_M.copy(), index=periods_M)
    endog_date_M = pd.DataFrame(dta_M.copy(), index=dates_M)
    nobs_Q = 3
    dates_Q = pd.date_range(start='2000', periods=nobs_Q, freq=freq_Q)
    periods_Q = pd.period_range(start='2000', periods=nobs_Q, freq='Q')
    dta_Q = np.random.normal(size=(nobs_Q, 2))
    endog_period_Q = pd.DataFrame(dta_Q.copy(), index=periods_Q)
    endog_date_Q = pd.DataFrame(dta_Q.copy(), index=dates_Q)
    mod_base = dynamic_factor_mq.DynamicFactorMQ(endog_period_M, endog_quarterly=endog_period_Q)
    mod = dynamic_factor_mq.DynamicFactorMQ(endog_date_M, endog_quarterly=endog_date_Q)
    assert_(mod._index.equals(mod_base._index))
    assert_allclose(mod.endog, mod_base.endog)
    mod = dynamic_factor_mq.DynamicFactorMQ(endog_date_M, endog_quarterly=endog_period_Q)
    assert_(mod._index.equals(mod_base._index))
    assert_allclose(mod.endog, mod_base.endog)
    mod = dynamic_factor_mq.DynamicFactorMQ(endog_period_M, endog_quarterly=endog_date_Q)
    assert_(mod._index.equals(mod_base._index))
    assert_allclose(mod.endog, mod_base.endog)