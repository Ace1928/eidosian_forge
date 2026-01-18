import os
import warnings
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (
def test_arima000():
    np.random.seed(328423)
    nobs = 50
    endog = pd.DataFrame(np.random.normal(size=nobs))
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), measurement_error=False)
    res = mod.smooth(mod.start_params)
    assert_allclose(res.smoothed_state, endog.T)
    mod = sarimax.SARIMAX(endog, order=(0, 1, 0), measurement_error=False)
    res = mod.smooth(mod.start_params)
    assert_allclose(res.smoothed_state[1:, 1:], endog.diff()[1:].T)
    error = np.random.normal(size=nobs)
    endog = np.ones(nobs) * 10 + error
    exog = np.ones(nobs)
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), exog=exog)
    mod.ssm.filter_univariate = True
    res = mod.smooth([10.0, 1.0])
    assert_allclose(res.smoothed_state[0], error, atol=1e-10)
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), exog=exog, mle_regression=False)
    mod.ssm.filter_univariate = True
    mod.initialize_known([0.0, 10.0], np.diag([1.0, 0.0]))
    res = mod.smooth([1.0])
    assert_allclose(res.smoothed_state[0], error, atol=1e-10)
    assert_allclose(res.smoothed_state[1], 10, atol=1e-10)
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), exog=exog, mle_regression=False, time_varying_regression=True)
    mod.ssm.filter_univariate = True
    mod.initialize_known([10.0], np.diag([0.0]))
    res = mod.smooth([0.0, 1.0])
    assert_allclose(res.smoothed_state[0], 10, atol=1e-10)