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
@pytest.mark.parametrize('demean', [True, False])
@pytest.mark.parametrize('adjusted', [True, False])
@pytest.mark.parametrize('fft', [True, False])
def test_compare_acovf_vs_ccovf(demean, adjusted, fft, reset_randomstate):
    x = np.random.normal(size=128)
    F1 = acovf(x, demean=demean, adjusted=adjusted, fft=fft)
    F2 = ccovf(x, x, demean=demean, adjusted=adjusted, fft=fft)
    assert_almost_equal(F1, F2, decimal=7)