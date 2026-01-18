import numpy as np
import pytest
from pandas.compat import IS64
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('skipna', [True, False])
@pytest.mark.parametrize('method', ['min', 'max'])
def test_floating_array_min_max(skipna, method, dtype):
    arr = pd.array([0.0, 1.0, None], dtype=dtype)
    func = getattr(arr, method)
    result = func(skipna=skipna)
    if skipna:
        assert result == (0 if method == 'min' else 1)
    else:
        assert result is pd.NA