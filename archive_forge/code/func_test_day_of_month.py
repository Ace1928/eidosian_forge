from __future__ import annotations
from datetime import datetime
import pytest
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
from pandas import (
from pandas.tests.tseries.offsets.common import (
def test_day_of_month(self):
    dt = datetime(2007, 1, 1)
    offset = MonthEnd()
    result = dt + offset
    assert result == Timestamp(2007, 1, 31)
    result = result + offset
    assert result == Timestamp(2007, 2, 28)