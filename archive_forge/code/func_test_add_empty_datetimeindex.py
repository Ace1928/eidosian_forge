from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import PerformanceWarning
from pandas import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import WeekDay
from pandas.tseries import offsets
from pandas.tseries.offsets import (
def test_add_empty_datetimeindex(self, offset_types, tz_naive_fixture):
    offset_s = _create_offset(offset_types)
    dti = DatetimeIndex([], tz=tz_naive_fixture).as_unit('ns')
    warn = None
    if isinstance(offset_s, (Easter, WeekOfMonth, LastWeekOfMonth, CustomBusinessDay, BusinessHour, CustomBusinessHour, CustomBusinessMonthBegin, CustomBusinessMonthEnd, FY5253, FY5253Quarter)):
        warn = PerformanceWarning
    check_stacklevel = tz_naive_fixture is None
    with tm.assert_produces_warning(warn, check_stacklevel=check_stacklevel):
        result = dti + offset_s
    tm.assert_index_equal(result, dti)
    with tm.assert_produces_warning(warn, check_stacklevel=check_stacklevel):
        result = offset_s + dti
    tm.assert_index_equal(result, dti)
    dta = dti._data
    with tm.assert_produces_warning(warn, check_stacklevel=check_stacklevel):
        result = dta + offset_s
    tm.assert_equal(result, dta)
    with tm.assert_produces_warning(warn, check_stacklevel=check_stacklevel):
        result = offset_s + dta
    tm.assert_equal(result, dta)