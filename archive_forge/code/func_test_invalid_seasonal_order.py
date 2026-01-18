import os
import warnings
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (
def test_invalid_seasonal_order():
    endog = np.zeros(10)
    with pytest.raises(ValueError):
        sarimax.SARIMAX(endog, seasonal_order=(1,))
    with pytest.raises(ValueError):
        sarimax.SARIMAX(endog, seasonal_order=(1, 2, 3, 4, 5))
    with pytest.raises(ValueError):
        sarimax.SARIMAX(endog, seasonal_order=(1, 0, 0, 0))
    with pytest.raises(ValueError):
        sarimax.SARIMAX(endog, seasonal_order=(0, 0, 1, 0))
    with pytest.raises(ValueError):
        sarimax.SARIMAX(endog, seasonal_order=(1, 0, 1, 0))
    with pytest.raises(ValueError):
        sarimax.SARIMAX(endog, seasonal_order=(0, 0, 0, 1))