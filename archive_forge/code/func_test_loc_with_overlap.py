import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
def test_loc_with_overlap(self, indexer_sl):
    idx = IntervalIndex.from_tuples([(1, 5), (3, 7)])
    ser = Series(range(len(idx)), index=idx)
    expected = ser
    result = indexer_sl(ser)[4]
    tm.assert_series_equal(expected, result)
    result = indexer_sl(ser)[[4]]
    tm.assert_series_equal(expected, result)
    expected = 0
    result = indexer_sl(ser)[Interval(1, 5)]
    assert expected == result
    expected = ser
    result = indexer_sl(ser)[[Interval(1, 5), Interval(3, 7)]]
    tm.assert_series_equal(expected, result)
    with pytest.raises(KeyError, match=re.escape("Interval(3, 5, closed='right')")):
        indexer_sl(ser)[Interval(3, 5)]
    msg = "None of \\[IntervalIndex\\(\\[\\(3, 5\\]\\], dtype='interval\\[int64, right\\]'\\)\\] are in the \\[index\\]"
    with pytest.raises(KeyError, match=msg):
        indexer_sl(ser)[[Interval(3, 5)]]
    expected = ser
    result = indexer_sl(ser)[Interval(1, 5):Interval(3, 7)]
    tm.assert_series_equal(expected, result)
    msg = "'can only get slices from an IntervalIndex if bounds are non-overlapping and all monotonic increasing or decreasing'"
    with pytest.raises(KeyError, match=msg):
        indexer_sl(ser)[Interval(1, 6):Interval(3, 8)]
    if indexer_sl is tm.loc:
        with pytest.raises(KeyError, match=msg):
            ser.loc[1:4]