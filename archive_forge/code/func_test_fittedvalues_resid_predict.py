import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose, assert_
from statsmodels.datasets import macrodata
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.kalman_filter import (
@pytest.mark.parametrize('conserve_memory', [MEMORY_CONSERVE, MEMORY_NO_FORECAST_COV])
def test_fittedvalues_resid_predict(conserve_memory):
    endog = dta['infl'].iloc[:20]
    mod1 = sarimax.SARIMAX(endog, order=(1, 0, 0), concentrate_scale=True)
    mod2 = sarimax.SARIMAX(endog, order=(1, 0, 0), concentrate_scale=True)
    mod1.ssm.set_conserve_memory(conserve_memory)
    assert_equal(mod1.ssm.conserve_memory, conserve_memory)
    assert_equal(mod2.ssm.conserve_memory, 0)
    res1 = mod1.filter([0])
    res2 = mod2.filter([0])
    assert_equal(res1.filter_results.conserve_memory, conserve_memory | MEMORY_NO_SMOOTHING)
    assert_equal(res2.filter_results.conserve_memory, MEMORY_NO_SMOOTHING)
    assert_allclose(res1.fittedvalues, 0)
    assert_allclose(res1.predict(), 0)
    assert_allclose(res1.predict(start=endog.index[10]), np.zeros(10))
    assert_allclose(res1.resid, endog)
    assert_allclose(res1.forecast(3), np.zeros(3))
    assert_allclose(res1.fittedvalues, res2.fittedvalues)
    assert_allclose(res1.predict(), res2.predict())
    assert_allclose(res1.predict(start=endog.index[10]), res2.predict(start=endog.index[10]))
    assert_allclose(res1.resid, res2.resid)
    assert_allclose(res1.forecast(3), res2.forecast(3))
    assert_allclose(res1.test_normality('jarquebera'), res2.test_normality('jarquebera'))
    assert_allclose(res1.test_heteroskedasticity('breakvar'), res2.test_heteroskedasticity('breakvar'))
    actual = res1.test_serial_correlation('ljungbox')
    desired = res2.test_serial_correlation('ljungbox')
    assert_allclose(actual, desired)