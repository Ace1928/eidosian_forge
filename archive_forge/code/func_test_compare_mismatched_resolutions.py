from __future__ import annotations
from datetime import timedelta
import operator
import numpy as np
import pytest
from pandas._libs.tslibs import tz_compare
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_compare_mismatched_resolutions(self, comparison_op):
    op = comparison_op
    iinfo = np.iinfo(np.int64)
    vals = np.array([iinfo.min, iinfo.min + 1, iinfo.max], dtype=np.int64)
    arr = np.array(vals).view('M8[ns]')
    arr2 = arr.view('M8[s]')
    left = DatetimeArray._simple_new(arr, dtype=arr.dtype)
    right = DatetimeArray._simple_new(arr2, dtype=arr2.dtype)
    if comparison_op is operator.eq:
        expected = np.array([False, False, False])
    elif comparison_op is operator.ne:
        expected = np.array([True, True, True])
    elif comparison_op in [operator.lt, operator.le]:
        expected = np.array([False, False, True])
    else:
        expected = np.array([False, True, False])
    result = op(left, right)
    tm.assert_numpy_array_equal(result, expected)
    result = op(left[1], right)
    tm.assert_numpy_array_equal(result, expected)
    if op not in [operator.eq, operator.ne]:
        np_res = op(left._ndarray, right._ndarray)
        tm.assert_numpy_array_equal(np_res[1:], ~expected[1:])