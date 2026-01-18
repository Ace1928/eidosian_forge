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
def test_constructor_fromisocalendar(self):
    expected_timestamp = Timestamp('2000-01-03 00:00:00')
    expected_stdlib = datetime.fromisocalendar(2000, 1, 1)
    result = Timestamp.fromisocalendar(2000, 1, 1)
    assert result == expected_timestamp
    assert result == expected_stdlib
    assert isinstance(result, Timestamp)