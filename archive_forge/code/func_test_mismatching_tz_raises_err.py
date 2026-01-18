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
@pytest.mark.parametrize('start,end', [(Timestamp(dt1, tz=tz1), Timestamp(dt2)), (Timestamp(dt1), Timestamp(dt2, tz=tz2)), (Timestamp(dt1, tz=tz1), Timestamp(dt2, tz=tz2)), (Timestamp(dt1, tz=tz2), Timestamp(dt2, tz=tz1))])
def test_mismatching_tz_raises_err(self, start, end):
    msg = 'Start and end cannot both be tz-aware with different timezones'
    with pytest.raises(TypeError, match=msg):
        date_range(start, end)
    with pytest.raises(TypeError, match=msg):
        date_range(start, end, freq=BDay())