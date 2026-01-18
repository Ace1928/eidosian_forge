from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo
def test_ar6_no_quarterly(reset_randomstate):
    ix = pd.period_range(start='1950-01', periods=1, freq='M')
    faux = pd.Series([0], index=ix)
    mod = sarimax.SARIMAX(faux, order=(6, 0, 0))
    params = np.r_[0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0]
    endog = mod.simulate(params, 100)
    mod_ar = sarimax.SARIMAX(endog, order=(6, 0, 0))
    mod_dfm = dynamic_factor_mq.DynamicFactorMQ(endog, factor_orders=6, standardize=False, idiosyncratic_ar1=False)
    llf_ar = mod_ar.loglike(params)
    llf_dfm = mod_dfm.loglike(np.r_[1, params, 0.0])
    assert_allclose(llf_dfm, llf_ar)
    if not SKIP_MONTE_CARLO_TESTS:
        res_dfm = mod_dfm.fit()
        actual = res_dfm.params
        actual[-2] *= actual[0]
        actual[0] = 1
        assert_allclose(res_dfm.params[1:-1], params, atol=0.01)