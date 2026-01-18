import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
@pytest.mark.parametrize('values, expected', [([1, 2, 3], 6), ([1, 2, 3, None], 6), ([None], 0)])
def test_integer_array_numpy_sum(values, expected):
    arr = pd.array(values, dtype='Int64')
    result = np.sum(arr)
    assert result == expected