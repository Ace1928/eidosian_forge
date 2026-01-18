from collections import deque
import re
import string
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray
@pytest.mark.parametrize('flip', [True, False])
def test_binary_ufunc_scalar(ufunc, sparse, flip, arrays_for_binary_ufunc):
    arr, _ = arrays_for_binary_ufunc
    if sparse:
        arr = SparseArray(arr)
    other = 2
    series = pd.Series(arr, name='name')
    series_args = (series, other)
    array_args = (arr, other)
    if flip:
        series_args = tuple(reversed(series_args))
        array_args = tuple(reversed(array_args))
    expected = pd.Series(ufunc(*array_args), name='name')
    result = ufunc(*series_args)
    tm.assert_series_equal(result, expected)