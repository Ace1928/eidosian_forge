from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
from pandas.tests.tseries.offsets.common import assert_offset_equal
from pandas.tseries.holiday import USFederalHolidayCalendar
@pytest.fixture
def offset1():
    return CustomBusinessHour(weekmask='Tue Wed Thu Fri')