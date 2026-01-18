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
def test_innovations_algo_brockwell_davis():
    ma = -0.9
    acovf = np.array([1 + ma ** 2, ma])
    theta, sigma2 = innovations_algo(acovf, nobs=4)
    exp_theta = np.array([[0], [-0.4972], [-0.6606], [-0.7404]])
    assert_allclose(theta, exp_theta, rtol=0.0001)
    assert_allclose(sigma2, [1.81, 1.3625, 1.2155, 1.1436], rtol=0.0001)
    theta, sigma2 = innovations_algo(acovf, nobs=500)
    assert_allclose(theta[-1, 0], ma)
    assert_allclose(sigma2[-1], 1.0)