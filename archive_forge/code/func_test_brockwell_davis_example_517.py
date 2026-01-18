import numpy as np
import pytest
from numpy.testing import assert_allclose
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake
from statsmodels.tsa.arima.estimators.hannan_rissanen import (
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tools.tools import Bunch
@pytest.mark.low_precision('Test against Example 5.1.7 in Brockwell and Davis (2016)')
def test_brockwell_davis_example_517():
    endog = lake.copy()
    hr, _ = hannan_rissanen(endog, ar_order=1, ma_order=1, demean=True, initial_ar_order=22, unbiased=False)
    assert_allclose(hr.ar_params, [0.6961], atol=0.0001)
    assert_allclose(hr.ma_params, [0.3788], atol=0.0001)
    u, v = arma_innovations(endog - endog.mean(), hr.ar_params, hr.ma_params, sigma2=1)
    tmp = u / v ** 0.5
    assert_allclose(np.inner(tmp, tmp) / len(u), 0.4774, atol=0.0001)