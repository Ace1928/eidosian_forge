import decimal
import numpy as np
import pytest
from pandas.core.dtypes.cast import maybe_downcast_to_dtype
from pandas import (
import pandas._testing as tm
def test_downcast_conversion_nan(float_numpy_dtype):
    dtype = float_numpy_dtype
    data = [1.0, 2.0, np.nan]
    expected = np.array(data, dtype=dtype)
    arr = np.array(data, dtype=dtype)
    result = maybe_downcast_to_dtype(arr, 'infer')
    tm.assert_almost_equal(result, expected)