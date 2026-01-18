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
def test_innovations_mle():
    endog = dta['infl'].iloc[:100]
    desired_p, _ = innovations_mle(endog, order=(1, 0, 1), demean=False)
    mod = ARIMA(endog, order=(1, 0, 1), trend='n')
    res = mod.fit(method='innovations_mle')
    assert_allclose(res.params, desired_p.params, atol=1e-05)
    desired_p, _ = innovations_mle(endog, order=(1, 0, 0), seasonal_order=(1, 0, 0, 4), demean=False)
    mod = ARIMA(endog, order=(1, 0, 0), seasonal_order=(1, 0, 0, 4), trend='n')
    res = mod.fit(method='innovations_mle')
    assert_allclose(res.params, desired_p.params, atol=1e-05)