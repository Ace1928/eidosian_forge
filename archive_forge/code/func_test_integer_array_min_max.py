import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
@pytest.mark.parametrize('skipna', [True, False])
@pytest.mark.parametrize('method', ['min', 'max'])
def test_integer_array_min_max(skipna, method, any_int_ea_dtype):
    dtype = any_int_ea_dtype
    arr = pd.array([0, 1, None], dtype=dtype)
    func = getattr(arr, method)
    result = func(skipna=skipna)
    if skipna:
        assert result == (0 if method == 'min' else 1)
    else:
        assert result is pd.NA