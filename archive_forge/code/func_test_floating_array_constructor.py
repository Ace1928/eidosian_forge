import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
from pandas.core.arrays.floating import (
def test_floating_array_constructor():
    values = np.array([1, 2, 3, 4], dtype='float64')
    mask = np.array([False, False, False, True], dtype='bool')
    result = FloatingArray(values, mask)
    expected = pd.array([1, 2, 3, np.nan], dtype='Float64')
    tm.assert_extension_array_equal(result, expected)
    tm.assert_numpy_array_equal(result._data, values)
    tm.assert_numpy_array_equal(result._mask, mask)
    msg = ".* should be .* numpy array. Use the 'pd.array' function instead"
    with pytest.raises(TypeError, match=msg):
        FloatingArray(values.tolist(), mask)
    with pytest.raises(TypeError, match=msg):
        FloatingArray(values, mask.tolist())
    with pytest.raises(TypeError, match=msg):
        FloatingArray(values.astype(int), mask)
    msg = "__init__\\(\\) missing 1 required positional argument: 'mask'"
    with pytest.raises(TypeError, match=msg):
        FloatingArray(values)