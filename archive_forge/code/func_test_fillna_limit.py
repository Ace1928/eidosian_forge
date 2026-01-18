from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_fillna_limit(self):
    ser = Series(np.nan, index=[0, 1, 2])
    result = ser.fillna(999, limit=1)
    expected = Series([999, np.nan, np.nan], index=[0, 1, 2])
    tm.assert_series_equal(result, expected)
    result = ser.fillna(999, limit=2)
    expected = Series([999, 999, np.nan], index=[0, 1, 2])
    tm.assert_series_equal(result, expected)