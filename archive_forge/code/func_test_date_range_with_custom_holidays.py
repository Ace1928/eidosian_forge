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
def test_date_range_with_custom_holidays(self, unit):
    freq = offsets.CustomBusinessHour(start='15:00', holidays=['2020-11-26'])
    result = date_range(start='2020-11-25 15:00', periods=4, freq=freq, unit=unit)
    expected = DatetimeIndex(['2020-11-25 15:00:00', '2020-11-25 16:00:00', '2020-11-27 15:00:00', '2020-11-27 16:00:00'], dtype=f'M8[{unit}]', freq=freq)
    tm.assert_index_equal(result, expected)