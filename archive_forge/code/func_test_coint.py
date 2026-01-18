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
def test_coint():
    nobs = 200
    scale_e = 1
    const = [1, 0, 0.5, 0]
    np.random.seed(123)
    unit = np.random.randn(nobs).cumsum()
    y = scale_e * np.random.randn(nobs, 4)
    y[:, :2] += unit[:, None]
    y += const
    y = np.round(y, 4)
    for trend in []:
        print('\n', trend)
        print(coint(y[:, 0], y[:, 1], trend=trend, maxlag=4, autolag=None))
        print(coint(y[:, 0], y[:, 1:3], trend=trend, maxlag=4, autolag=None))
        print(coint(y[:, 0], y[:, 2:], trend=trend, maxlag=4, autolag=None))
        print(coint(y[:, 0], y[:, 1:], trend=trend, maxlag=4, autolag=None))
    res_egranger = {}
    res = res_egranger['ct'] = {}
    res[0] = [-5.615251442239, -4.406102369132, -3.82866685109, -3.532082997903]
    res[1] = [-5.63591313706, -4.758609717199, -4.179130554708, -3.880909696863]
    res[2] = [-2.892029275027, -4.758609717199, -4.179130554708, -3.880909696863]
    res[3] = [-5.626932544079, -5.08363327039, -4.502469783057, -4.2031051091]
    res = res_egranger['c'] = {}
    res[0] = [-5.760696844656, -3.952043522638, -3.367006313729, -3.065831247948]
    res[0][1] = -3.952321293401682
    res[1] = [-5.781087068772, -4.367111915942, -3.783961136005, -3.483501524709]
    res[2] = [-2.477444137366, -4.367111915942, -3.783961136005, -3.483501524709]
    res[3] = [-5.778205811661, -4.735249216434, -4.152738973763, -3.852480848968]
    res = res_egranger['ctt'] = {}
    res[0] = [-5.644431269946, -4.796038299708, -4.221469431008, -3.926472577178]
    res[1] = [-5.665691609506, -5.111158174219, -4.53317278104, -4.23601008516]
    res[2] = [-3.161462374828, -5.111158174219, -4.53317278104, -4.23601008516]
    res[3] = [-5.657904558563, -5.406880189412, -4.826111619543, -4.527090164875]
    res = res_egranger['n'] = {}
    nan = np.nan
    res[0] = [-3.7146175989071137, nan, nan, nan]
    res[1] = [-3.8199323012888384, nan, nan, nan]
    res[2] = [-1.686500079127068, nan, nan, nan]
    res[3] = [-3.7991270451873675, nan, nan, nan]
    for trend in ['c', 'ct', 'ctt', 'n']:
        res1 = {}
        res1[0] = coint(y[:, 0], y[:, 1], trend=trend, maxlag=4, autolag=None)
        res1[1] = coint(y[:, 0], y[:, 1:3], trend=trend, maxlag=4, autolag=None)
        res1[2] = coint(y[:, 0], y[:, 2:], trend=trend, maxlag=4, autolag=None)
        res1[3] = coint(y[:, 0], y[:, 1:], trend=trend, maxlag=4, autolag=None)
        for i in range(4):
            res = res_egranger[trend]
            assert_allclose(res1[i][0], res[i][0], rtol=1e-11)
            r2 = res[i][1:]
            r1 = res1[i][2]
            assert_allclose(r1, r2, rtol=0, atol=6e-07)
    res1_0 = coint(y[:, 0], y[:, 1], trend='ct', maxlag=4)
    assert_allclose(res1_0[2], res_egranger['ct'][0][1:], rtol=0, atol=6e-07)
    assert_allclose(res1_0[:2], [-13.992946638547112, 2.270898990540678e-27], rtol=1e-10, atol=1e-27)