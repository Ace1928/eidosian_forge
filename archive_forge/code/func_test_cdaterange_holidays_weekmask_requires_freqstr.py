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
def test_cdaterange_holidays_weekmask_requires_freqstr(self):
    msg = 'a custom frequency string is required when holidays or weekmask are passed, got frequency B'
    with pytest.raises(ValueError, match=msg):
        bdate_range('2013-05-01', periods=3, weekmask='Sun Mon Tue Wed Thu', holidays=['2013-05-01'])