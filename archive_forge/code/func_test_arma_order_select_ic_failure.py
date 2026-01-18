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
def test_arma_order_select_ic_failure():
    y = np.array([0.8607437781720364, 0.8531654906790692, 0.871046537743633, 0.6069238206898739, 0.6922594196730131, 0.7333617724890934, 0.03661329261479619, 0.1569306723996238, 0.12777403512447857, -0.27531446294481976, -0.2419813963165358, -0.2390331795123639, -0.260002413259065, -0.21282920015519238, -0.15943768324388355, 0.2516930156426878, 0.17623057091518773, 0.1267813336879139, 0.8975582908675317, 0.8266706879535015])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = arma_order_select_ic(y)