from statsmodels.compat.platform import PLATFORM_WIN32
import io
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose, assert_raises, assert_
from statsmodels.datasets import macrodata
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.estimators.yule_walker import yule_walker
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
from statsmodels.tsa.arima.estimators.statespace import statespace
def test_forecast_with_exog():
    endog = dta['infl'].iloc[:100].values
    exog = np.arange(len(endog)) ** 2
    mod = ARIMA(endog[:50], order=(1, 1, 0), exog=exog[:50], trend='t')
    res = mod.filter([0.2, 0.05, 0.3, 1.0])
    endog2 = endog.copy()
    endog2[50:] = np.nan
    mod2 = mod.clone(endog2, exog=exog)
    print(mod.param_names)
    print(mod2.param_names)
    res2 = mod2.filter(res.params)
    assert_allclose(res.forecast(50, exog=exog[50:]), res2.fittedvalues[-50:])