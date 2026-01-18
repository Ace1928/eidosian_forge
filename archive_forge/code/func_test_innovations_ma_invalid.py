import numpy as np
import pytest
from numpy.testing import (
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import (
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
def test_innovations_ma_invalid():
    endog = np.arange(2)
    assert_raises(ValueError, innovations, endog, ma_order=2)
    assert_raises(ValueError, innovations, endog, ma_order=-1)
    assert_raises(ValueError, innovations, endog, ma_order=1.5)
    endog = np.arange(10)
    assert_raises(ValueError, innovations, endog, ma_order=[1, 3])