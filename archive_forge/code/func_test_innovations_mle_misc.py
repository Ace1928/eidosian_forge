import numpy as np
import pytest
from numpy.testing import (
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import (
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
def test_innovations_mle_misc():
    endog = np.arange(20) ** 2 * 1.0
    hr, _ = hannan_rissanen(endog, ar_order=1, demean=False)
    assert_(hr.ar_params[0] > 1)
    _, res = innovations_mle(endog, order=(1, 0, 0))
    assert_allclose(res.start_params[0], 0)
    hr, _ = hannan_rissanen(endog, ma_order=1, demean=False)
    assert_(hr.ma_params[0] > 1)
    _, res = innovations_mle(endog, order=(0, 0, 1))
    assert_allclose(res.start_params[0], 0)