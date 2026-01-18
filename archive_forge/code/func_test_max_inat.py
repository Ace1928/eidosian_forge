import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_max_inat():
    ser = Series([1, iNaT])
    key = np.array([1, 1], dtype=np.int64)
    gb = ser.groupby(key)
    result = gb.max(min_count=2)
    expected = Series({1: 1}, dtype=np.int64)
    tm.assert_series_equal(result, expected, check_exact=True)
    result = gb.min(min_count=2)
    expected = Series({1: iNaT}, dtype=np.int64)
    tm.assert_series_equal(result, expected, check_exact=True)
    result = gb.min(min_count=3)
    expected = Series({1: np.nan})
    tm.assert_series_equal(result, expected, check_exact=True)