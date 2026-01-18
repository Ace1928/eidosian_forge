from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_fillna_nonscalar(self):
    s1 = Series([np.nan])
    s2 = Series([1])
    result = s1.fillna(s2)
    expected = Series([1.0])
    tm.assert_series_equal(result, expected)
    result = s1.fillna({})
    tm.assert_series_equal(result, s1)
    result = s1.fillna(Series((), dtype=object))
    tm.assert_series_equal(result, s1)
    result = s2.fillna(s1)
    tm.assert_series_equal(result, s2)
    result = s1.fillna({0: 1})
    tm.assert_series_equal(result, expected)
    result = s1.fillna({1: 1})
    tm.assert_series_equal(result, Series([np.nan]))
    result = s1.fillna({0: 1, 1: 1})
    tm.assert_series_equal(result, expected)
    result = s1.fillna(Series({0: 1, 1: 1}))
    tm.assert_series_equal(result, expected)
    result = s1.fillna(Series({0: 1, 1: 1}, index=[4, 5]))
    tm.assert_series_equal(result, s1)