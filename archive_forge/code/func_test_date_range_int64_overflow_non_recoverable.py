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
def test_date_range_int64_overflow_non_recoverable(self):
    msg = 'Cannot generate range with'
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        date_range(start='1970-02-01', periods=106752 * 24, freq='h')
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        date_range(end='1969-11-14', periods=106752 * 24, freq='h')