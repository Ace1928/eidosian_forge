from datetime import timedelta
import numpy as np
import pytest
from pandas.core.dtypes.common import is_integer
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import Day
def test_float_freq(self):
    result = interval_range(0, 1, freq=0.1)
    expected = IntervalIndex.from_breaks([0 + 0.1 * n for n in range(11)])
    tm.assert_index_equal(result, expected)
    result = interval_range(0, 1, freq=0.6)
    expected = IntervalIndex.from_breaks([0, 0.6])
    tm.assert_index_equal(result, expected)