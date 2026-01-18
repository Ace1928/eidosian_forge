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
@pytest.mark.parametrize('start', ['is_month_start', 'is_quarter_start', 'is_year_start'])
@pytest.mark.parametrize('tz', [None, 'US/Eastern'])
def test_is_start(self, start, tz):
    ts = Timestamp('2014-01-01 00:00:00', tz=tz)
    assert getattr(ts, start)