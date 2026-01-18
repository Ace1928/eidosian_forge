from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import MONTH_END, YEAR_END, assert_index_equal
from statsmodels.compat.platform import PLATFORM_WIN
from statsmodels.compat.python import lrange
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas import DataFrame, Series, date_range
import pytest
from scipy import stats
from scipy.interpolate import interp1d
from statsmodels.datasets import macrodata, modechoice, nile, randhie, sunspots
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import array_like, bool_like
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import (
def test_acovf2d(reset_randomstate):
    dta = sunspots.load_pandas().data
    dta.index = date_range(start='1700', end='2009', freq=YEAR_END)[:309]
    del dta['YEAR']
    res = acovf(dta, fft=False)
    assert_equal(res, acovf(dta.values, fft=False))
    x = np.random.random((10, 2))
    with pytest.raises(ValueError):
        acovf(x, fft=False)