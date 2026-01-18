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
def test_constructor_subclassed_datetime(self):

    class SubDatetime(datetime):
        pass
    data = SubDatetime(2000, 1, 1)
    result = Timestamp(data)
    expected = Timestamp(2000, 1, 1)
    assert result == expected