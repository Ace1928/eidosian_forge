import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import FloatingArray
def test_bitwise(dtype):
    left = pd.array([1, None, 3, 4], dtype=dtype)
    right = pd.array([None, 3, 5, 4], dtype=dtype)
    result = left | right
    expected = pd.array([None, None, 3 | 5, 4 | 4], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)
    result = left & right
    expected = pd.array([None, None, 3 & 5, 4 & 4], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)
    result = left ^ right
    expected = pd.array([None, None, 3 ^ 5, 4 ^ 4], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)
    floats = right.astype('Float64')
    with pytest.raises(TypeError, match='unsupported operand type'):
        left | floats
    with pytest.raises(TypeError, match='unsupported operand type'):
        left & floats
    with pytest.raises(TypeError, match='unsupported operand type'):
        left ^ floats