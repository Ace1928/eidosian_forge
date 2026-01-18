import numpy as np
import pytest
from numpy.testing import (
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake, oshorts
from statsmodels.tsa.arima.estimators.gls import gls
@pytest.mark.low_precision('Test against Example 6.6.2 in Brockwell and Davis (2016)')
def test_brockwell_davis_example_662():
    endog = lake.copy()
    exog = np.c_[np.ones_like(endog), np.arange(1, len(endog) + 1) * 1.0]
    res, _ = gls(endog, exog, order=(2, 0, 0))
    assert_allclose(res.exog_params, [10.091, -0.0216], atol=0.001)
    assert_allclose(res.ar_params, [1.005, -0.291], atol=0.001)
    assert_allclose(res.sigma2, 0.4571, atol=0.001)