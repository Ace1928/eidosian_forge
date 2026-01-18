import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_max_inat_not_all_na():
    ser = Series([1, iNaT, 2, iNaT + 1])
    gb = ser.groupby([1, 2, 3, 3])
    result = gb.min(min_count=2)
    expected = Series({1: np.nan, 2: np.nan, 3: iNaT + 1})
    expected.index = expected.index.astype(np.int_)
    tm.assert_series_equal(result, expected, check_exact=True)