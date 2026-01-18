from itertools import permutations
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('breaks', [date_range('20180101', periods=4), date_range('20180101', periods=4, tz='US/Eastern'), timedelta_range('0 days', periods=4)], ids=lambda x: str(x.dtype))
def test_maybe_convert_i8(self, breaks):
    index = IntervalIndex.from_breaks(breaks)
    result = index._maybe_convert_i8(index)
    expected = IntervalIndex.from_breaks(breaks.asi8)
    tm.assert_index_equal(result, expected)
    interval = Interval(breaks[0], breaks[1])
    result = index._maybe_convert_i8(interval)
    expected = Interval(breaks[0]._value, breaks[1]._value)
    assert result == expected
    result = index._maybe_convert_i8(breaks)
    expected = Index(breaks.asi8)
    tm.assert_index_equal(result, expected)
    result = index._maybe_convert_i8(breaks[0])
    expected = breaks[0]._value
    assert result == expected
    result = index._maybe_convert_i8(list(breaks))
    expected = Index(breaks.asi8)
    tm.assert_index_equal(result, expected)