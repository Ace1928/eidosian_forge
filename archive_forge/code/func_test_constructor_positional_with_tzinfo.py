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
def test_constructor_positional_with_tzinfo(self):
    ts = Timestamp(2020, 12, 31, tzinfo=timezone.utc)
    expected = Timestamp('2020-12-31', tzinfo=timezone.utc)
    assert ts == expected