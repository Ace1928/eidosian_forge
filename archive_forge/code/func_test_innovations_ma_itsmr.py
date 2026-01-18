import numpy as np
import pytest
from numpy.testing import (
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import (
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
def test_innovations_ma_itsmr():
    endog = lake.copy()
    check_innovations_ma_itsmr(endog)
    check_innovations_ma_itsmr(endog.values)
    check_innovations_ma_itsmr(endog.tolist())