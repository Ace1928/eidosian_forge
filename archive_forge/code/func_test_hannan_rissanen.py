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
def test_hannan_rissanen():
    endog = dta['infl'].diff().iloc[1:101]
    desired_p, _ = hannan_rissanen(endog, ar_order=1, ma_order=1, demean=False)
    mod = ARIMA(endog, order=(1, 0, 1), trend='n')
    res = mod.fit(method='hannan_rissanen')
    assert_allclose(res.params, desired_p.params)