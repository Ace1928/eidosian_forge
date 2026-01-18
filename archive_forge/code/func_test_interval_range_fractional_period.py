from datetime import timedelta
import numpy as np
import pytest
from pandas.core.dtypes.common import is_integer
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import Day
def test_interval_range_fractional_period(self):
    expected = interval_range(start=0, periods=10)
    msg = "Non-integer 'periods' in pd.date_range, .* pd.interval_range"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = interval_range(start=0, periods=10.5)
    tm.assert_index_equal(result, expected)