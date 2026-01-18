import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import algorithms
from pandas.core.arrays import PeriodArray
def test_isin_with_i8(self):
    expected = Series([True, True, False, False, False])
    expected2 = Series([False, True, False, False, False])
    s = Series(date_range('jan-01-2013', 'jan-05-2013'))
    result = s.isin(s[0:2])
    tm.assert_series_equal(result, expected)
    result = s.isin(s[0:2].values)
    tm.assert_series_equal(result, expected)
    result = s.isin([s[1]])
    tm.assert_series_equal(result, expected2)
    result = s.isin([np.datetime64(s[1])])
    tm.assert_series_equal(result, expected2)
    result = s.isin(set(s[0:2]))
    tm.assert_series_equal(result, expected)
    s = Series(pd.to_timedelta(range(5), unit='d'))
    result = s.isin(s[0:2])
    tm.assert_series_equal(result, expected)