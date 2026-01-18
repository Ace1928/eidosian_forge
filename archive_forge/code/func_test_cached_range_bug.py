from datetime import (
import re
import numpy as np
import pytest
import pytz
from pytz import timezone
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.offsets import (
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.datetimes import _generate_range as generate_range
from pandas.tests.indexes.datetimes.test_timezones import (
from pandas.tseries.holiday import USFederalHolidayCalendar
def test_cached_range_bug(self):
    rng = date_range('2010-09-01 05:00:00', periods=50, freq=DateOffset(hours=6))
    assert len(rng) == 50
    assert rng[0] == datetime(2010, 9, 1, 5)