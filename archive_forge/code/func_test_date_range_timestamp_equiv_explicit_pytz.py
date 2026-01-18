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
def test_date_range_timestamp_equiv_explicit_pytz(self):
    rng = date_range('20090415', '20090519', tz=pytz.timezone('US/Eastern'))
    stamp = rng[0]
    ts = Timestamp('20090415', tz=pytz.timezone('US/Eastern'))
    assert ts == stamp