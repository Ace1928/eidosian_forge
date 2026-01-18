import calendar
from datetime import (
import locale
import time
import unicodedata
from dateutil.tz import (
from hypothesis import (
import numpy as np
import pytest
import pytz
from pytz import utc
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.timezones import (
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
def test_addsub_offset(self, ts_tz):
    off = offsets.YearEnd(1)
    result = ts_tz + off
    assert isinstance(result, Timestamp)
    assert result._creso == ts_tz._creso
    if ts_tz.month == 12 and ts_tz.day == 31:
        assert result.year == ts_tz.year + 1
    else:
        assert result.year == ts_tz.year
    assert result.day == 31
    assert result.month == 12
    assert tz_compare(result.tz, ts_tz.tz)
    result = ts_tz - off
    assert isinstance(result, Timestamp)
    assert result._creso == ts_tz._creso
    assert result.year == ts_tz.year - 1
    assert result.day == 31
    assert result.month == 12
    assert tz_compare(result.tz, ts_tz.tz)