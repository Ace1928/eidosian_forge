from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_loc_with_values_including_missing_values(self):
    idx = MultiIndex.from_product([[np.nan, 1]] * 2)
    expected = slice(0, 2, None)
    assert idx.get_loc(np.nan) == expected
    idx = MultiIndex.from_arrays([[np.nan, 1, 2, np.nan]])
    expected = np.array([True, False, False, True])
    tm.assert_numpy_array_equal(idx.get_loc(np.nan), expected)
    idx = MultiIndex.from_product([[np.nan, 1]] * 3)
    expected = slice(2, 4, None)
    assert idx.get_loc((np.nan, 1)) == expected