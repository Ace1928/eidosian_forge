import numpy as np
import pytest
from numpy.testing import (
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake, oshorts
from statsmodels.tsa.arima.estimators.gls import gls
@pytest.mark.low_precision('Test against Example 6.6.1 in Brockwell and Davis (2016)')
def test_brockwell_davis_example_661():
    endog = oshorts.copy()
    exog = np.ones_like(endog)
    res, _ = gls(endog, exog, order=(0, 0, 1), max_iter=1, tolerance=1)
    assert_allclose(res.exog_params, -4.745, atol=0.001)
    assert_allclose(res.ma_params, -0.818, atol=0.001)
    assert_allclose(res.sigma2, 2041, atol=1)
    res, _ = gls(endog, exog, order=(0, 0, 1))
    assert_allclose(res.exog_params, -4.78, atol=0.001)
    assert_allclose(res.ma_params, -0.848, atol=0.001)