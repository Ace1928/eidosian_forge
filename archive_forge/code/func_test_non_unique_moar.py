import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
def test_non_unique_moar(self, indexer_sl):
    idx = IntervalIndex.from_tuples([(1, 3), (1, 3), (3, 7)])
    ser = Series(range(len(idx)), index=idx)
    expected = ser.iloc[[0, 1]]
    result = indexer_sl(ser)[Interval(1, 3)]
    tm.assert_series_equal(expected, result)
    expected = ser
    result = indexer_sl(ser)[Interval(1, 3):]
    tm.assert_series_equal(expected, result)
    expected = ser.iloc[[0, 1]]
    result = indexer_sl(ser)[[Interval(1, 3)]]
    tm.assert_series_equal(expected, result)