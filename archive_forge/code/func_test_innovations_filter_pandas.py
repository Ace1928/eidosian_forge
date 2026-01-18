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
def test_innovations_filter_pandas(reset_randomstate):
    ma = np.array([-0.9, 0.5])
    acovf = np.array([1 + (ma ** 2).sum(), ma[0] + ma[1] * ma[0], ma[1]])
    theta, _ = innovations_algo(acovf, nobs=10)
    endog = np.random.randn(10)
    endog_pd = pd.Series(endog, index=pd.date_range('2000-01-01', periods=10))
    resid = innovations_filter(endog, theta)
    resid_pd = innovations_filter(endog_pd, theta)
    assert_allclose(resid, resid_pd.values)
    assert_index_equal(endog_pd.index, resid_pd.index)