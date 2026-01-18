from datetime import (
import re
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.parsing import DateParseError
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('bound, offset', [(Timestamp.min, -1), (Timestamp.max, 1)])
@pytest.mark.parametrize('period_property', ['start_time', 'end_time'])
def test_inner_bounds_start_and_end_time(self, bound, offset, period_property):
    period = TestPeriodProperties._period_constructor(bound, -offset)
    expected = period.to_timestamp().round(freq='s')
    assert getattr(period, period_property).round(freq='s') == expected
    expected = (bound - offset * Timedelta(1, unit='s')).floor('s')
    assert getattr(period, period_property).floor('s') == expected