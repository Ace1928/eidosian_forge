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
def test_low_memory():
    endog = dta['infl'].iloc[:50]
    mod = ARIMA(endog, order=(1, 0, 0), concentrate_scale=True)
    res1 = mod.fit()
    res2 = mod.fit(low_memory=True)
    assert_allclose(res2.params, res1.params)
    assert_allclose(res2.llf, res1.llf)
    assert_equal(mod.ssm.memory_conserve, 0)
    assert_(res2.llf_obs is None)
    assert_(res2.predicted_state is None)
    assert_(res2.filtered_state is None)
    assert_(res2.smoothed_state is None)