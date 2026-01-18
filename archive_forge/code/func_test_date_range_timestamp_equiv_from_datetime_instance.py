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
def test_date_range_timestamp_equiv_from_datetime_instance(self):
    datetime_instance = datetime(2014, 3, 4)
    timestamp_instance = date_range(datetime_instance, periods=1, freq='D')[0]
    ts = Timestamp(datetime_instance)
    assert ts == timestamp_instance