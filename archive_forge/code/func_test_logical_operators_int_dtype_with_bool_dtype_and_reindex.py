from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_logical_operators_int_dtype_with_bool_dtype_and_reindex(self):
    index = list('bca')
    s_tft = Series([True, False, True], index=index)
    s_tft = Series([True, False, True], index=index)
    s_tff = Series([True, False, False], index=index)
    s_0123 = Series(range(4), dtype='int64')
    expected = Series([False] * 7, index=[0, 1, 2, 3, 'a', 'b', 'c'])
    with tm.assert_produces_warning(FutureWarning):
        result = s_tft & s_0123
    tm.assert_series_equal(result, expected)
    expected = Series([False] * 7, index=[0, 1, 2, 3, 'a', 'b', 'c'])
    with tm.assert_produces_warning(FutureWarning):
        result = s_0123 & s_tft
    tm.assert_series_equal(result, expected)
    s_a0b1c0 = Series([1], list('b'))
    with tm.assert_produces_warning(FutureWarning):
        res = s_tft & s_a0b1c0
    expected = s_tff.reindex(list('abc'))
    tm.assert_series_equal(res, expected)
    with tm.assert_produces_warning(FutureWarning):
        res = s_tft | s_a0b1c0
    expected = s_tft.reindex(list('abc'))
    tm.assert_series_equal(res, expected)