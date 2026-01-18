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
def test_non_nano_fields(self, dt64, ts):
    alt = Timestamp(dt64)
    assert ts.year == alt.year
    assert ts.month == alt.month
    assert ts.day == alt.day
    assert ts.hour == ts.minute == ts.second == ts.microsecond == 0
    assert ts.nanosecond == 0
    assert ts.to_julian_date() == alt.to_julian_date()
    assert ts.weekday() == alt.weekday()
    assert ts.isoweekday() == alt.isoweekday()