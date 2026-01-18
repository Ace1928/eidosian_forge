from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_compare_td64_ndarray(self):
    arr = np.arange(5).astype('timedelta64[ns]')
    td = Timedelta(arr[1])
    expected = np.array([False, True, False, False, False], dtype=bool)
    result = td == arr
    tm.assert_numpy_array_equal(result, expected)
    result = arr == td
    tm.assert_numpy_array_equal(result, expected)
    result = td != arr
    tm.assert_numpy_array_equal(result, ~expected)
    result = arr != td
    tm.assert_numpy_array_equal(result, ~expected)