from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_find_nan(any_string_dtype):
    ser = Series(['ABCDEFG', np.nan, 'DEFGHIJEF', np.nan, 'XXXX'], dtype=any_string_dtype)
    expected_dtype = np.float64 if any_string_dtype in object_pyarrow_numpy else 'Int64'
    result = ser.str.find('EF')
    expected = Series([4, np.nan, 1, np.nan, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    result = ser.str.rfind('EF')
    expected = Series([4, np.nan, 7, np.nan, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    result = ser.str.find('EF', 3)
    expected = Series([4, np.nan, 7, np.nan, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    result = ser.str.rfind('EF', 3)
    expected = Series([4, np.nan, 7, np.nan, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    result = ser.str.find('EF', 3, 6)
    expected = Series([4, np.nan, -1, np.nan, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    result = ser.str.rfind('EF', 3, 6)
    expected = Series([4, np.nan, -1, np.nan, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)