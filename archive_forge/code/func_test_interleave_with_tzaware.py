import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_interleave_with_tzaware(self, timezone_frame):
    result = timezone_frame.assign(D='foo').values
    expected = np.array([[Timestamp('2013-01-01 00:00:00'), Timestamp('2013-01-02 00:00:00'), Timestamp('2013-01-03 00:00:00')], [Timestamp('2013-01-01 00:00:00-0500', tz='US/Eastern'), NaT, Timestamp('2013-01-03 00:00:00-0500', tz='US/Eastern')], [Timestamp('2013-01-01 00:00:00+0100', tz='CET'), NaT, Timestamp('2013-01-03 00:00:00+0100', tz='CET')], ['foo', 'foo', 'foo']], dtype=object).T
    tm.assert_numpy_array_equal(result, expected)
    result = timezone_frame.values
    expected = np.array([[Timestamp('2013-01-01 00:00:00'), Timestamp('2013-01-02 00:00:00'), Timestamp('2013-01-03 00:00:00')], [Timestamp('2013-01-01 00:00:00-0500', tz='US/Eastern'), NaT, Timestamp('2013-01-03 00:00:00-0500', tz='US/Eastern')], [Timestamp('2013-01-01 00:00:00+0100', tz='CET'), NaT, Timestamp('2013-01-03 00:00:00+0100', tz='CET')]], dtype=object).T
    tm.assert_numpy_array_equal(result, expected)