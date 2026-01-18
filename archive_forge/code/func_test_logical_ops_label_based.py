from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_logical_ops_label_based(self, using_infer_string):
    a = Series([True, False, True], list('bca'))
    b = Series([False, True, False], list('abc'))
    expected = Series([False, True, False], list('abc'))
    result = a & b
    tm.assert_series_equal(result, expected)
    expected = Series([True, True, False], list('abc'))
    result = a | b
    tm.assert_series_equal(result, expected)
    expected = Series([True, False, False], list('abc'))
    result = a ^ b
    tm.assert_series_equal(result, expected)
    a = Series([True, False, True], list('bca'))
    b = Series([False, True, False, True], list('abcd'))
    expected = Series([False, True, False, False], list('abcd'))
    result = a & b
    tm.assert_series_equal(result, expected)
    expected = Series([True, True, False, False], list('abcd'))
    result = a | b
    tm.assert_series_equal(result, expected)
    empty = Series([], dtype=object)
    result = a & empty.copy()
    expected = Series([False, False, False], list('abc'))
    tm.assert_series_equal(result, expected)
    result = a | empty.copy()
    expected = Series([True, True, False], list('abc'))
    tm.assert_series_equal(result, expected)
    with tm.assert_produces_warning(FutureWarning):
        result = a & Series([1], ['z'])
    expected = Series([False, False, False, False], list('abcz'))
    tm.assert_series_equal(result, expected)
    with tm.assert_produces_warning(FutureWarning):
        result = a | Series([1], ['z'])
    expected = Series([True, True, False, False], list('abcz'))
    tm.assert_series_equal(result, expected)
    with tm.assert_produces_warning(FutureWarning):
        for e in [empty.copy(), Series([1], ['z']), Series(np.nan, b.index), Series(np.nan, a.index)]:
            result = a[a | e]
            tm.assert_series_equal(result, a[a])
    for e in [Series(['z'])]:
        warn = FutureWarning if using_infer_string else None
        if using_infer_string:
            import pyarrow as pa
            with tm.assert_produces_warning(warn, match='Operation between non'):
                with pytest.raises(pa.lib.ArrowNotImplementedError, match='has no kernel'):
                    result = a[a | e]
        else:
            result = a[a | e]
        tm.assert_series_equal(result, a[a])
    index = list('bca')
    t = Series([True, False, True])
    for v in [True, 1, 2]:
        result = Series([True, False, True], index=index) | v
        expected = Series([True, True, True], index=index)
        tm.assert_series_equal(result, expected)
    msg = 'Cannot perform.+with a dtyped.+array and scalar of type'
    for v in [np.nan, 'foo']:
        with pytest.raises(TypeError, match=msg):
            t | v
    for v in [False, 0]:
        result = Series([True, False, True], index=index) | v
        expected = Series([True, False, True], index=index)
        tm.assert_series_equal(result, expected)
    for v in [True, 1]:
        result = Series([True, False, True], index=index) & v
        expected = Series([True, False, True], index=index)
        tm.assert_series_equal(result, expected)
    for v in [False, 0]:
        result = Series([True, False, True], index=index) & v
        expected = Series([False, False, False], index=index)
        tm.assert_series_equal(result, expected)
    msg = 'Cannot perform.+with a dtyped.+array and scalar of type'
    for v in [np.nan]:
        with pytest.raises(TypeError, match=msg):
            t & v