import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_raises
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import dowj, lake
from statsmodels.tsa.arima.estimators.durbin_levinson import durbin_levinson
def test_itsmr():
    endog = lake.copy()
    check_itsmr(endog)
    check_itsmr(endog.values)
    check_itsmr(endog.tolist())