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
@pytest.mark.parametrize('tz, option, expected', [['US/Pacific', 'shift_forward', '2019-03-10 03:00'], ['dateutil/US/Pacific', 'shift_forward', '2019-03-10 03:00'], ['US/Pacific', 'shift_backward', '2019-03-10 01:00'], ['dateutil/US/Pacific', 'shift_backward', '2019-03-10 01:00'], ['US/Pacific', timedelta(hours=1), '2019-03-10 03:00']])
def test_date_range_nonexistent_endpoint(self, tz, option, expected):
    with pytest.raises(pytz.NonExistentTimeError, match='2019-03-10 02:00:00'):
        date_range('2019-03-10 00:00', '2019-03-10 02:00', tz='US/Pacific', freq='h')
    times = date_range('2019-03-10 00:00', '2019-03-10 02:00', freq='h', tz=tz, nonexistent=option)
    assert times[-1] == Timestamp(expected, tz=tz)