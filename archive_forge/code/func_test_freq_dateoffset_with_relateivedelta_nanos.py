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
def test_freq_dateoffset_with_relateivedelta_nanos(self):
    freq = DateOffset(hours=10, days=57, nanoseconds=3)
    result = date_range(end='1970-01-01 00:00:00', periods=10, freq=freq, name='a')
    expected = DatetimeIndex(['1968-08-02T05:59:59.999999973', '1968-09-28T15:59:59.999999976', '1968-11-25T01:59:59.999999979', '1969-01-21T11:59:59.999999982', '1969-03-19T21:59:59.999999985', '1969-05-16T07:59:59.999999988', '1969-07-12T17:59:59.999999991', '1969-09-08T03:59:59.999999994', '1969-11-04T13:59:59.999999997', '1970-01-01T00:00:00.000000000'], name='a')
    tm.assert_index_equal(result, expected)