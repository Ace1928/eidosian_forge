from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.timedeltas import TimedeltaArray
def test_float64_ns_rounded(self):
    tdi = TimedeltaIndex([2.3, 9.7])
    expected = TimedeltaIndex([2, 9])
    tm.assert_index_equal(tdi, expected)
    tdi = TimedeltaIndex([2.0, 9.0])
    expected = TimedeltaIndex([2, 9])
    tm.assert_index_equal(tdi, expected)
    tdi = TimedeltaIndex([2.0, np.nan])
    expected = TimedeltaIndex([Timedelta(nanoseconds=2), pd.NaT])
    tm.assert_index_equal(tdi, expected)