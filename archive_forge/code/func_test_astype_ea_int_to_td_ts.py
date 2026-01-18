import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_astype_ea_int_to_td_ts():
    ser = Series([1, None], dtype='Int64')
    result = ser.astype('m8[ns]')
    expected = Series([1, Timedelta('nat')], dtype='m8[ns]')
    tm.assert_series_equal(result, expected)
    result = ser.astype('M8[ns]')
    expected = Series([1, Timedelta('nat')], dtype='M8[ns]')
    tm.assert_series_equal(result, expected)