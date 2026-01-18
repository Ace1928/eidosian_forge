import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('arrays', [(date_range('20180101', periods=4), date_range('20180103', periods=4)), (date_range('20180101', periods=4, tz='US/Eastern'), date_range('20180103', periods=4, tz='US/Eastern')), (timedelta_range('0 days', periods=4), timedelta_range('2 days', periods=4))], ids=lambda x: str(x[0].dtype))
def test_get_loc_datetimelike_overlapping(self, arrays):
    index = IntervalIndex.from_arrays(*arrays)
    value = index[0].mid + Timedelta('12 hours')
    result = index.get_loc(value)
    expected = slice(0, 2, None)
    assert result == expected
    interval = Interval(index[0].left, index[0].right)
    result = index.get_loc(interval)
    expected = 0
    assert result == expected