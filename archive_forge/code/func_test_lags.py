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
def test_lags(self):
    with pytest.warns(InterpolationWarning):
        res = kpss(self.x, 'c', nlags='auto')
    assert_equal(res[2], 9)
    res = kpss(sunspots.load().data['SUNACTIVITY'], 'c', nlags='auto')
    assert_equal(res[2], 7)
    with pytest.warns(InterpolationWarning):
        res = kpss(nile.load().data['volume'], 'c', nlags='auto')
    assert_equal(res[2], 5)
    with pytest.warns(InterpolationWarning):
        res = kpss(randhie.load().data['lncoins'], 'ct', nlags='auto')
    assert_equal(res[2], 75)
    with pytest.warns(InterpolationWarning):
        res = kpss(modechoice.load().data['invt'], 'ct', nlags='auto')
    assert_equal(res[2], 18)