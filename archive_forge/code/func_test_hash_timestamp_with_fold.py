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
@pytest.mark.parametrize('timezone, year, month, day, hour', [['America/Chicago', 2013, 11, 3, 1], ['America/Santiago', 2021, 4, 3, 23]])
def test_hash_timestamp_with_fold(self, timezone, year, month, day, hour):
    test_timezone = gettz(timezone)
    transition_1 = Timestamp(year=year, month=month, day=day, hour=hour, minute=0, fold=0, tzinfo=test_timezone)
    transition_2 = Timestamp(year=year, month=month, day=day, hour=hour, minute=0, fold=1, tzinfo=test_timezone)
    assert hash(transition_1) == hash(transition_2)