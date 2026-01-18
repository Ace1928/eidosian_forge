import re
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_indexing_with_datetimeindex_tz(self, indexer_sl):
    index = date_range('2015-01-01', periods=2, tz='utc')
    ser = Series(range(2), index=index, dtype='int64')
    for sel in (index, list(index)):
        result = indexer_sl(ser)[sel]
        expected = ser.copy()
        if sel is not index:
            expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(result, expected)
        result = ser.copy()
        indexer_sl(result)[sel] = 1
        expected = Series(1, index=index)
        tm.assert_series_equal(result, expected)
    assert indexer_sl(ser)[index[1]] == 1
    result = ser.copy()
    indexer_sl(result)[index[1]] = 5
    expected = Series([0, 5], index=index)
    tm.assert_series_equal(result, expected)