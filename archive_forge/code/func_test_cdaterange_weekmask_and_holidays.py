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
def test_cdaterange_weekmask_and_holidays(self, unit):
    result = bdate_range('2013-05-01', periods=3, freq='C', weekmask='Sun Mon Tue Wed Thu', holidays=['2013-05-01'], unit=unit)
    expected = DatetimeIndex(['2013-05-02', '2013-05-05', '2013-05-06'], dtype=f'M8[{unit}]', freq=result.freq)
    tm.assert_index_equal(result, expected)
    assert result.freq == expected.freq