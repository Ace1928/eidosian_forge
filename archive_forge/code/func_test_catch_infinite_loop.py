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
def test_catch_infinite_loop(self):
    offset = offsets.DateOffset(minute=5)
    msg = 'Offset <DateOffset: minute=5> did not increment date'
    with pytest.raises(ValueError, match=msg):
        date_range(datetime(2011, 11, 11), datetime(2011, 11, 12), freq=offset)