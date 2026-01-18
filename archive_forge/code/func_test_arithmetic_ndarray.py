from datetime import (
import pickle
import numpy as np
import pytest
from pandas._libs.missing import NA
from pandas.core.dtypes.common import is_scalar
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('shape', [(3,), (3, 3), (1, 2, 3)])
def test_arithmetic_ndarray(shape, all_arithmetic_functions):
    op = all_arithmetic_functions
    a = np.zeros(shape)
    if op.__name__ == 'pow':
        a += 5
    result = op(NA, a)
    expected = np.full(a.shape, NA, dtype=object)
    tm.assert_numpy_array_equal(result, expected)