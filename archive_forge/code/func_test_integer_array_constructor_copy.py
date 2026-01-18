import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_integer
from pandas.core.arrays import IntegerArray
from pandas.core.arrays.integer import (
def test_integer_array_constructor_copy():
    values = np.array([1, 2, 3, 4], dtype='int64')
    mask = np.array([False, False, False, True], dtype='bool')
    result = IntegerArray(values, mask)
    assert result._data is values
    assert result._mask is mask
    result = IntegerArray(values, mask, copy=True)
    assert result._data is not values
    assert result._mask is not mask