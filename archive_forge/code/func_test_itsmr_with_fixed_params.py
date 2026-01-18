import numpy as np
import pytest
from numpy.testing import assert_allclose
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake
from statsmodels.tsa.arima.estimators.hannan_rissanen import (
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tools.tools import Bunch
@pytest.mark.parametrize('fixed_params', [{'ar.L1': 0.69607715}, {'ma.L1': 0.37879692}, {'ar.L1': 0.69607715, 'ma.L1': 0.37879692}])
def test_itsmr_with_fixed_params(fixed_params):
    endog = lake.copy()
    hr, _ = hannan_rissanen(endog, ar_order=1, ma_order=1, demean=True, initial_ar_order=22, unbiased=False, fixed_params=fixed_params)
    assert_allclose(hr.ar_params, [0.69607715], atol=0.0001)
    assert_allclose(hr.ma_params, [0.3787969217], atol=0.0001)
    u, v = arma_innovations(endog - endog.mean(), hr.ar_params, hr.ma_params, sigma2=1)
    tmp = u / v ** 0.5
    assert_allclose(np.inner(tmp, tmp) / len(u), 0.4773580109, atol=0.0001)