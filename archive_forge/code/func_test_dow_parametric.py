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
@given(ts=st.datetimes(), sign=st.sampled_from(['-', '']))
def test_dow_parametric(self, ts, sign):
    ts = f'{sign}{str(ts.year).zfill(4)}-{str(ts.month).zfill(2)}-{str(ts.day).zfill(2)}'
    result = Timestamp(ts).weekday()
    expected = ((np.datetime64(ts) - np.datetime64('1970-01-01')).astype('int64') - 4) % 7
    assert result == expected