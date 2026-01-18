import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_interpolate_datetimelike_values(self, frame_or_series):
    orig = Series(date_range('2012-01-01', periods=5))
    ser = orig.copy()
    ser[2] = NaT
    res = frame_or_series(ser).interpolate()
    expected = frame_or_series(orig)
    tm.assert_equal(res, expected)
    ser_tz = ser.dt.tz_localize('US/Pacific')
    res_tz = frame_or_series(ser_tz).interpolate()
    expected_tz = frame_or_series(orig.dt.tz_localize('US/Pacific'))
    tm.assert_equal(res_tz, expected_tz)
    ser_td = ser - ser[0]
    res_td = frame_or_series(ser_td).interpolate()
    expected_td = frame_or_series(orig - orig[0])
    tm.assert_equal(res_td, expected_td)