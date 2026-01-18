from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_fillna_downcast_infer_objects_to_numeric(self):
    arr = np.arange(5).astype(object)
    arr[3] = np.nan
    ser = Series(arr)
    msg = "The 'downcast' keyword in fillna is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = ser.fillna(3, downcast='infer')
    expected = Series(np.arange(5), dtype=np.int64)
    tm.assert_series_equal(res, expected)
    msg = "The 'downcast' keyword in ffill is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = ser.ffill(downcast='infer')
    expected = Series([0, 1, 2, 2, 4], dtype=np.int64)
    tm.assert_series_equal(res, expected)
    msg = "The 'downcast' keyword in bfill is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = ser.bfill(downcast='infer')
    expected = Series([0, 1, 2, 4, 4], dtype=np.int64)
    tm.assert_series_equal(res, expected)
    ser[2] = 2.5
    expected = Series([0, 1, 2.5, 3, 4], dtype=np.float64)
    msg = "The 'downcast' keyword in fillna is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = ser.fillna(3, downcast='infer')
    tm.assert_series_equal(res, expected)
    msg = "The 'downcast' keyword in ffill is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = ser.ffill(downcast='infer')
    expected = Series([0, 1, 2.5, 2.5, 4], dtype=np.float64)
    tm.assert_series_equal(res, expected)
    msg = "The 'downcast' keyword in bfill is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = ser.bfill(downcast='infer')
    expected = Series([0, 1, 2.5, 4, 4], dtype=np.float64)
    tm.assert_series_equal(res, expected)