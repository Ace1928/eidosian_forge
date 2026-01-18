from datetime import (
from hypothesis import given
import numpy as np
import pytest
from pandas._libs.tslibs import ccalendar
from pandas._testing._hypothesis import DATETIME_IN_PD_TIMESTAMP_RANGE_NO_TZ
def test_get_day_of_year_dt():
    dt = datetime.fromordinal(1 + np.random.default_rng(2).integers(365 * 4000))
    result = ccalendar.get_day_of_year(dt.year, dt.month, dt.day)
    expected = (dt - dt.replace(month=1, day=1)).days + 1
    assert result == expected