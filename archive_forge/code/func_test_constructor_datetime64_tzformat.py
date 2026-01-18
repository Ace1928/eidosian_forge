from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
@pytest.mark.parametrize('freq', ['YS', 'W-SUN'])
def test_constructor_datetime64_tzformat(self, freq):
    idx = date_range('2013-01-01T00:00:00-05:00', '2016-01-01T23:59:59-05:00', freq=freq)
    expected = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59', freq=freq, tz=timezone(timedelta(minutes=-300)))
    tm.assert_index_equal(idx, expected)
    expected_i8 = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59', freq=freq, tz='America/Lima')
    tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)
    idx = date_range('2013-01-01T00:00:00+09:00', '2016-01-01T23:59:59+09:00', freq=freq)
    expected = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59', freq=freq, tz=timezone(timedelta(minutes=540)))
    tm.assert_index_equal(idx, expected)
    expected_i8 = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59', freq=freq, tz='Asia/Tokyo')
    tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)
    idx = date_range('2013/1/1 0:00:00-5:00', '2016/1/1 23:59:59-5:00', freq=freq)
    expected = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59', freq=freq, tz=timezone(timedelta(minutes=-300)))
    tm.assert_index_equal(idx, expected)
    expected_i8 = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59', freq=freq, tz='America/Lima')
    tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)
    idx = date_range('2013/1/1 0:00:00+9:00', '2016/1/1 23:59:59+09:00', freq=freq)
    expected = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59', freq=freq, tz=timezone(timedelta(minutes=540)))
    tm.assert_index_equal(idx, expected)
    expected_i8 = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59', freq=freq, tz='Asia/Tokyo')
    tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)