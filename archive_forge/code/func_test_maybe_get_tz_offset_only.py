from datetime import (
import dateutil.tz
import pytest
import pytz
from pandas._libs.tslibs import (
from pandas.compat import is_platform_windows
from pandas import Timestamp
def test_maybe_get_tz_offset_only():
    tz = timezones.maybe_get_tz(timezone.utc)
    assert tz == timezone(timedelta(hours=0, minutes=0))
    tz = timezones.maybe_get_tz('+01:15')
    assert tz == timezone(timedelta(hours=1, minutes=15))
    tz = timezones.maybe_get_tz('-01:15')
    assert tz == timezone(-timedelta(hours=1, minutes=15))
    tz = timezones.maybe_get_tz('UTC+02:45')
    assert tz == timezone(timedelta(hours=2, minutes=45))
    tz = timezones.maybe_get_tz('UTC-02:45')
    assert tz == timezone(-timedelta(hours=2, minutes=45))