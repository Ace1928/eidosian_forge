import decimal
import numpy as np
import pytest
from pandas.core.dtypes.cast import maybe_downcast_to_dtype
from pandas import (
import pandas._testing as tm
def test_downcast_conversion_no_nan(any_real_numpy_dtype):
    dtype = any_real_numpy_dtype
    expected = np.array([1, 2])
    arr = np.array([1.0, 2.0], dtype=dtype)
    result = maybe_downcast_to_dtype(arr, 'infer')
    tm.assert_almost_equal(result, expected, check_dtype=False)