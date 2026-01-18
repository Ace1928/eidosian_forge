from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries import offsets
def test_holidays(self):
    holidays = ['2012-01-31', datetime(2012, 2, 28), np.datetime64('2012-02-29')]
    bm_offset = CBMonthEnd(holidays=holidays)
    dt = datetime(2012, 1, 1)
    assert dt + bm_offset == datetime(2012, 1, 30)
    assert dt + 2 * bm_offset == datetime(2012, 2, 27)