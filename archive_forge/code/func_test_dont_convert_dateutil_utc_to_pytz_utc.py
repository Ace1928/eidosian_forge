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
def test_dont_convert_dateutil_utc_to_pytz_utc(self):
    result = Timestamp(datetime(2018, 1, 1), tz=tzutc())
    expected = Timestamp(datetime(2018, 1, 1)).tz_localize(tzutc())
    assert result == expected