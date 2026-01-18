from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_fillna_consistency(self):
    ser = Series([Timestamp('20130101'), NaT])
    result = ser.fillna(Timestamp('20130101', tz='US/Eastern'))
    expected = Series([Timestamp('20130101'), Timestamp('2013-01-01', tz='US/Eastern')], dtype='object')
    tm.assert_series_equal(result, expected)
    result = ser.where([True, False], Timestamp('20130101', tz='US/Eastern'))
    tm.assert_series_equal(result, expected)
    result = ser.where([True, False], Timestamp('20130101', tz='US/Eastern'))
    tm.assert_series_equal(result, expected)
    result = ser.fillna('foo')
    expected = Series([Timestamp('20130101'), 'foo'])
    tm.assert_series_equal(result, expected)
    ser2 = ser.copy()
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        ser2[1] = 'foo'
    tm.assert_series_equal(ser2, expected)