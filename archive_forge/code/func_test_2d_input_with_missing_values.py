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
def test_2d_input_with_missing_values(self):
    input_residuals = np.array([[0.0, 0.0, np.nan], [1.0, np.nan, 1.0], [2.0, 2.0, np.nan], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0], [8.0, 8.0, 8.0]])
    expected_statistic = np.array([(8.0 ** 2 + 7.0 ** 2 + 6.0 ** 2) / (0.0 ** 2 + 1.0 ** 2 + 2.0 ** 2), (8.0 ** 2 + 7.0 ** 2 + 6.0 ** 2) / (0.0 ** 2 + 2.0 ** 2), np.nan])
    expected_pvalue = np.array([2 * min(self.f.cdf(expected_statistic[0], 3, 3), self.f.sf(expected_statistic[0], 3, 3)), 2 * min(self.f.cdf(expected_statistic[1], 3, 2), self.f.sf(expected_statistic[1], 3, 2)), np.nan])
    actual_statistic, actual_pvalue = breakvar_heteroskedasticity_test(input_residuals)
    assert_equal(actual_statistic, expected_statistic)
    assert_equal(actual_pvalue, expected_pvalue)