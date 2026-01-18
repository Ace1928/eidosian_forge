from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo
def test_invalid_model_specification():
    dta = np.zeros((10, 2))
    dta[0] = 1.0
    dta_pd = pd.DataFrame(dta)
    dta_period_W = pd.DataFrame(dta, index=pd.period_range(start='2000', periods=10, freq='W'))
    dta_date_W = pd.DataFrame(dta, index=pd.date_range(start='2000', periods=10, freq='W'))
    dta_period_M = pd.DataFrame(dta, index=pd.period_range(start='2000', periods=10, freq='M'))
    dta_date_M = pd.DataFrame(dta, index=pd.date_range(start='2000', periods=10, freq=MONTH_END))
    dta_period_Q = pd.DataFrame(dta, index=pd.period_range(start='2000', periods=10, freq='Q'))
    msg = 'The model must contain at least one factor.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta, factors=0)
    msg = '`factors` argument must an integer number of factors, a list of global factor names, or a dictionary, mapping observed variables to factors.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta, factors=True)
    msg = '`factor_orders` argument must either be an integer or a dictionary.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta, factor_orders=True)
    msg = '`factor_multiplicities` argument must either be an integer or a dictionary.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta, factor_multiplicities=True)
    msg = f'Number of factors \\({dta.shape[1] + 1}\\) cannot be greater than'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta, factors=dta.shape[1] + 1)
    factor_orders = {('a', 'b'): 1, 'b': 2}
    msg = 'Each factor can be assigned to at most one block of factors in `factor_orders`.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta, factors=['a', 'b'], factor_orders=factor_orders)
    msg = 'If `endog_quarterly` is specified, then `endog` must contain only monthly variables, and so `k_endog_monthly` cannot be specified since it will be inferred from the shape of `endog`.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta_period_M, k_endog_monthly=2, endog_quarterly=dta)
    msg = 'Invalid value passed for `standardize`.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta_period_M, standardize='a')
    msg = 'If a `factors` dictionary is provided, then it must include entries for each observed variable.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta, factors={'y1': ['a']})
    msg = 'Each observed variable must be mapped to at least one factor in the `factors` dictionary.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta, factors={'y1': ['a'], 'y2': []})
    msg = 'Constant variable\\(s\\) found in observed variables, but constants cannot be included in this model.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta * 0)
    msg = 'Given monthly dataset is not a Pandas object.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta, endog_quarterly=dta)
    msg = 'Given quarterly dataset is not a Pandas object.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta_period_M, endog_quarterly=dta)
    msg = 'Given monthly dataset has an index with non-date values.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta_pd, endog_quarterly=dta_period_Q)
    msg = 'Given quarterly dataset has an index with non-date values.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta_period_M, endog_quarterly=dta_pd)
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta_date_M, endog_quarterly=dta_pd)
    msg = 'Index of given monthly dataset has a non-monthly frequency'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta_period_W, endog_quarterly=dta_period_Q)
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta_date_W, endog_quarterly=dta_period_Q)
    msg = 'Index of given quarterly dataset has a non-quarterly frequency'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta_period_M, endog_quarterly=dta_period_W)
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta_date_M, endog_quarterly=dta_date_W)