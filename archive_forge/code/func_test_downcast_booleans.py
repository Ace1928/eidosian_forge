import decimal
import numpy as np
import pytest
from pandas.core.dtypes.cast import maybe_downcast_to_dtype
from pandas import (
import pandas._testing as tm
def test_downcast_booleans():
    ser = Series([True, True, False])
    result = maybe_downcast_to_dtype(ser, np.dtype(np.float64))
    expected = ser.values
    tm.assert_numpy_array_equal(result, expected)