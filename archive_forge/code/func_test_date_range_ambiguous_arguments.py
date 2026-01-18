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
def test_date_range_ambiguous_arguments(self):
    start = datetime(2011, 1, 1, 5, 3, 40)
    end = datetime(2011, 1, 1, 8, 9, 40)
    msg = 'Of the four parameters: start, end, periods, and freq, exactly three must be specified'
    with pytest.raises(ValueError, match=msg):
        date_range(start, end, periods=10, freq='s')