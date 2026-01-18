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
def test_bday_near_overflow(self):
    start = Timestamp.max.floor('D').to_pydatetime()
    rng = date_range(start, end=None, periods=1, freq='B')
    expected = DatetimeIndex([start], freq='B').as_unit('ns')
    tm.assert_index_equal(rng, expected)