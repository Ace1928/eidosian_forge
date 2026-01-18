from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_astype_datetime64tz(self):
    ser = Series(date_range('20130101', periods=3, tz='US/Eastern'))
    result = ser.astype(object)
    expected = Series(ser.astype(object), dtype=object)
    tm.assert_series_equal(result, expected)
    result = Series(ser.values).dt.tz_localize('UTC').dt.tz_convert(ser.dt.tz)
    tm.assert_series_equal(result, ser)
    result = Series(ser.astype(object))
    expected = ser.astype(object)
    tm.assert_series_equal(result, expected)
    msg = 'Cannot use .astype to convert from timezone-naive'
    with pytest.raises(TypeError, match=msg):
        Series(ser.values).astype('datetime64[ns, US/Eastern]')
    with pytest.raises(TypeError, match=msg):
        Series(ser.values).astype(ser.dtype)
    result = ser.astype('datetime64[ns, CET]')
    expected = Series(date_range('20130101 06:00:00', periods=3, tz='CET'))
    tm.assert_series_equal(result, expected)