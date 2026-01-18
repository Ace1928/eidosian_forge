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
def test_coint_identical_series():
    nobs = 200
    scale_e = 1
    np.random.seed(123)
    y = scale_e * np.random.randn(nobs)
    warnings.simplefilter('always', CollinearityWarning)
    with pytest.warns(CollinearityWarning):
        c = coint(y, y, trend='c', maxlag=0, autolag=None)
    assert_equal(c[1], 0.0)
    assert_(np.isneginf(c[0]))