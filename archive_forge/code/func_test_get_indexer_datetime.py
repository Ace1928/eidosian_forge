import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
def test_get_indexer_datetime(self):
    ii = IntervalIndex.from_breaks(date_range('2018-01-01', periods=4))
    target = DatetimeIndex(['2018-01-02'], dtype='M8[ns]')
    result = ii.get_indexer(target)
    expected = np.array([0], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)
    result = ii.get_indexer(target.astype(str))
    tm.assert_numpy_array_equal(result, expected)
    result = ii.get_indexer(target.asi8)
    expected = np.array([-1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)