from collections import deque
import re
import string
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray
def test_multiple_output_ufunc(sparse, arrays_for_binary_ufunc):
    arr, _ = arrays_for_binary_ufunc
    if sparse:
        arr = SparseArray(arr)
    series = pd.Series(arr, name='name')
    result = np.modf(series)
    expected = np.modf(arr)
    assert isinstance(result, tuple)
    assert isinstance(expected, tuple)
    tm.assert_series_equal(result[0], pd.Series(expected[0], name='name'))
    tm.assert_series_equal(result[1], pd.Series(expected[1], name='name'))