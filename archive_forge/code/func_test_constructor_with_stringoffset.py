import calendar
from datetime import (
import zoneinfo
import dateutil.tz
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.compat import PY310
from pandas.errors import OutOfBoundsDatetime
from pandas import (
def test_constructor_with_stringoffset(self):
    base_str = '2014-07-01 11:00:00+02:00'
    base_dt = datetime(2014, 7, 1, 9)
    base_expected = 1404205200000000000
    assert calendar.timegm(base_dt.timetuple()) * 1000000000 == base_expected
    tests = [(base_str, base_expected), ('2014-07-01 12:00:00+02:00', base_expected + 3600 * 1000000000), ('2014-07-01 11:00:00.000008000+02:00', base_expected + 8000), ('2014-07-01 11:00:00.000000005+02:00', base_expected + 5)]
    timezones = [(None, 0), ('UTC', 0), (pytz.utc, 0), ('Asia/Tokyo', 9), ('US/Eastern', -4), ('dateutil/US/Pacific', -7), (pytz.FixedOffset(-180), -3), (dateutil.tz.tzoffset(None, 18000), 5)]
    for date_str, expected in tests:
        for result in [Timestamp(date_str)]:
            assert result.as_unit('ns')._value == expected
            result = Timestamp(result)
            assert result.as_unit('ns')._value == expected
        for tz, offset in timezones:
            result = Timestamp(date_str, tz=tz)
            expected_tz = expected
            assert result.as_unit('ns')._value == expected_tz
            result = Timestamp(result)
            assert result.as_unit('ns')._value == expected_tz
            result = Timestamp(result).tz_convert('UTC')
            expected_utc = expected
            assert result.as_unit('ns')._value == expected_utc
    result = Timestamp('2013-11-01 00:00:00-0500', tz='America/Chicago')
    assert result._value == Timestamp('2013-11-01 05:00')._value
    expected = "Timestamp('2013-11-01 00:00:00-0500', tz='America/Chicago')"
    assert repr(result) == expected
    assert result == eval(repr(result))
    result = Timestamp('2013-11-01 00:00:00-0500', tz='Asia/Tokyo')
    assert result._value == Timestamp('2013-11-01 05:00')._value
    expected = "Timestamp('2013-11-01 14:00:00+0900', tz='Asia/Tokyo')"
    assert repr(result) == expected
    assert result == eval(repr(result))
    result = Timestamp('2015-11-18 15:45:00+05:45', tz='Asia/Katmandu')
    assert result._value == Timestamp('2015-11-18 10:00')._value
    expected = "Timestamp('2015-11-18 15:45:00+0545', tz='Asia/Katmandu')"
    assert repr(result) == expected
    assert result == eval(repr(result))
    result = Timestamp('2015-11-18 15:30:00+05:30', tz='Asia/Kolkata')
    assert result._value == Timestamp('2015-11-18 10:00')._value
    expected = "Timestamp('2015-11-18 15:30:00+0530', tz='Asia/Kolkata')"
    assert repr(result) == expected
    assert result == eval(repr(result))