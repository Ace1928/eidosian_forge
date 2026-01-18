from statsmodels.compat.pandas import MONTH_END, QUARTER_END, YEAR_END
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import seasonal_decompose
def test_interpolate_trend(self):
    x = np.arange(12)
    freq = 4
    trend = seasonal_decompose(x, period=freq).trend
    assert_equal(trend[0], np.nan)
    trend = seasonal_decompose(x, period=freq, extrapolate_trend=5).trend
    assert_almost_equal(trend, x)
    trend = seasonal_decompose(x, period=freq, extrapolate_trend='freq').trend
    assert_almost_equal(trend, x)
    trend = seasonal_decompose(x[:, None], period=freq, extrapolate_trend=5).trend
    assert_almost_equal(trend, x)
    x = np.tile(np.arange(12), (2, 1)).T
    trend = seasonal_decompose(x, period=freq, extrapolate_trend=1).trend
    assert_almost_equal(trend, x)
    trend = seasonal_decompose(x, period=freq, extrapolate_trend='freq').trend
    assert_almost_equal(trend, x)