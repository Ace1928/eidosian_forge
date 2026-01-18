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
@pytest.mark.parametrize('td', [timedelta(days=4), Timedelta(days=4), np.timedelta64(4, 'D')])
def test_addsub_timedeltalike_non_nano(self, dt64, ts, td):
    exp_reso = max(ts._creso, Timedelta(td)._creso)
    result = ts - td
    expected = Timestamp(dt64) - td
    assert isinstance(result, Timestamp)
    assert result._creso == exp_reso
    assert result == expected
    result = ts + td
    expected = Timestamp(dt64) + td
    assert isinstance(result, Timestamp)
    assert result._creso == exp_reso
    assert result == expected
    result = td + ts
    expected = td + Timestamp(dt64)
    assert isinstance(result, Timestamp)
    assert result._creso == exp_reso
    assert result == expected