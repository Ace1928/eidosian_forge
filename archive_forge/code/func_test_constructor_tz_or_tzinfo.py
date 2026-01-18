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
def test_constructor_tz_or_tzinfo(self):
    stamps = [Timestamp(year=2017, month=10, day=22, tz='UTC'), Timestamp(year=2017, month=10, day=22, tzinfo=pytz.utc), Timestamp(year=2017, month=10, day=22, tz=pytz.utc), Timestamp(datetime(2017, 10, 22), tzinfo=pytz.utc), Timestamp(datetime(2017, 10, 22), tz='UTC'), Timestamp(datetime(2017, 10, 22), tz=pytz.utc)]
    assert all((ts == stamps[0] for ts in stamps))