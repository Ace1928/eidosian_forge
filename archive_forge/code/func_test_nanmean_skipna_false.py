from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@pytest.mark.parametrize('constructor', ['M8', 'm8'])
def test_nanmean_skipna_false(self, constructor, unit):
    dtype = f'{constructor}[{unit}]'
    arr = np.arange(12).astype(np.int64).view(dtype).reshape(4, 3)
    arr[-1, -1] = 'NaT'
    result = nanops.nanmean(arr, skipna=False)
    assert np.isnat(result)
    assert result.dtype == dtype
    result = nanops.nanmean(arr, axis=0, skipna=False)
    expected = np.array([4, 5, 'NaT'], dtype=arr.dtype)
    tm.assert_numpy_array_equal(result, expected)
    result = nanops.nanmean(arr, axis=1, skipna=False)
    expected = np.array([arr[0, 1], arr[1, 1], arr[2, 1], arr[-1, -1]])
    tm.assert_numpy_array_equal(result, expected)