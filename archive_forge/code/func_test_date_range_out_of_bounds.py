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
def test_date_range_out_of_bounds(self):
    msg = 'Cannot generate range'
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        date_range('2016-01-01', periods=100000, freq='D')
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        date_range(end='1763-10-12', periods=100000, freq='D')