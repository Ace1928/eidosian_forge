import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('breaks', [date_range('20180101', periods=4), date_range('20180101', periods=4, tz='US/Eastern'), timedelta_range('0 days', periods=4)], ids=lambda x: str(x.dtype))
def test_get_loc_datetimelike_nonoverlapping(self, breaks):
    index = IntervalIndex.from_breaks(breaks)
    value = index[0].mid
    result = index.get_loc(value)
    expected = 0
    assert result == expected
    interval = Interval(index[0].left, index[0].right)
    result = index.get_loc(interval)
    expected = 0
    assert result == expected