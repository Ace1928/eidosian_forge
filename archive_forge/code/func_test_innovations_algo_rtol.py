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
def test_innovations_algo_rtol():
    ma = np.array([-0.9, 0.5])
    acovf = np.array([1 + (ma ** 2).sum(), ma[0] + ma[1] * ma[0], ma[1]])
    theta, sigma2 = innovations_algo(acovf, nobs=500)
    theta_2, sigma2_2 = innovations_algo(acovf, nobs=500, rtol=1e-08)
    assert_allclose(theta, theta_2)
    assert_allclose(sigma2, sigma2_2)