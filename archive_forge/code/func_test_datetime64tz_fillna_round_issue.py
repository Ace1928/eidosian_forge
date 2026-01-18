from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_datetime64tz_fillna_round_issue(self):
    data = Series([NaT, NaT, datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=pytz.utc)])
    filled = data.bfill()
    expected = Series([datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=pytz.utc), datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=pytz.utc), datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=pytz.utc)])
    tm.assert_series_equal(filled, expected)