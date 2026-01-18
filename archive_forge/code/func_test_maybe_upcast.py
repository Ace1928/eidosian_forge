import numpy as np
import pytest
from pandas._libs.parsers import (
import pandas as pd
from pandas import NA
import pandas._testing as tm
from pandas.core.arrays import (
def test_maybe_upcast(any_real_numpy_dtype):
    dtype = np.dtype(any_real_numpy_dtype)
    na_value = na_values[dtype]
    arr = np.array([1, 2, na_value], dtype=dtype)
    result = _maybe_upcast(arr, use_dtype_backend=True)
    expected_mask = np.array([False, False, True])
    if issubclass(dtype.type, np.integer):
        expected = IntegerArray(arr, mask=expected_mask)
    else:
        expected = FloatingArray(arr, mask=expected_mask)
    tm.assert_extension_array_equal(result, expected)