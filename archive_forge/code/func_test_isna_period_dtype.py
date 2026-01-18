import numpy as np
from pandas import (
import pandas._testing as tm
def test_isna_period_dtype(self):
    ser = Series([Period('2011-01', freq='M'), Period('NaT', freq='M')])
    expected = Series([False, True])
    result = ser.isna()
    tm.assert_series_equal(result, expected)
    result = ser.notna()
    tm.assert_series_equal(result, ~expected)