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
def test_addition_doesnt_downcast_reso(self):
    ts = Timestamp(year=2022, month=1, day=1, microsecond=999999).as_unit('us')
    td = Timedelta(microseconds=1).as_unit('us')
    res = ts + td
    assert res._creso == ts._creso