import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
def test_non_unique(self, indexer_sl):
    idx = IntervalIndex.from_tuples([(1, 3), (3, 7)])
    ser = Series(range(len(idx)), index=idx)
    result = indexer_sl(ser)[Interval(1, 3)]
    assert result == 0
    result = indexer_sl(ser)[[Interval(1, 3)]]
    expected = ser.iloc[0:1]
    tm.assert_series_equal(expected, result)