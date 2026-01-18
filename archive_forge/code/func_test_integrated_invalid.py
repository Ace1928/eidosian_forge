import numpy as np
import pytest
from numpy.testing import (
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake, oshorts
from statsmodels.tsa.arima.estimators.gls import gls
def test_integrated_invalid():
    endog = lake.copy()
    exog = np.arange(1, len(endog) + 1) * 1.0
    assert_raises(ValueError, gls, endog, exog, order=(1, 1, 0), include_constant=True)