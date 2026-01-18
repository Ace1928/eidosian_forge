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
@pytest.mark.parametrize('date_string, expected', [('0000-2-29', 1), ('0000-3-1', 2), ('1582-10-14', 3), ('-0040-1-1', 4), ('2023-06-18', 6)])
def test_dow_historic(self, date_string, expected):
    ts = Timestamp(date_string)
    dow = ts.weekday()
    assert dow == expected