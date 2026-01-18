from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_logical_operators_int_dtype_with_int_dtype(self):
    s_0123 = Series(range(4), dtype='int64')
    s_3333 = Series([3] * 4)
    s_4444 = Series([4] * 4)
    res = s_0123 & s_3333
    expected = Series(range(4), dtype='int64')
    tm.assert_series_equal(res, expected)
    res = s_0123 | s_4444
    expected = Series(range(4, 8), dtype='int64')
    tm.assert_series_equal(res, expected)
    s_1111 = Series([1] * 4, dtype='int8')
    res = s_0123 & s_1111
    expected = Series([0, 1, 0, 1], dtype='int64')
    tm.assert_series_equal(res, expected)
    res = s_0123.astype(np.int16) | s_1111.astype(np.int32)
    expected = Series([1, 1, 3, 3], dtype='int32')
    tm.assert_series_equal(res, expected)