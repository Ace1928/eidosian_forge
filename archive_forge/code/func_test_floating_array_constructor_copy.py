import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
from pandas.core.arrays.floating import (
def test_floating_array_constructor_copy():
    values = np.array([1, 2, 3, 4], dtype='float64')
    mask = np.array([False, False, False, True], dtype='bool')
    result = FloatingArray(values, mask)
    assert result._data is values
    assert result._mask is mask
    result = FloatingArray(values, mask, copy=True)
    assert result._data is not values
    assert result._mask is not mask