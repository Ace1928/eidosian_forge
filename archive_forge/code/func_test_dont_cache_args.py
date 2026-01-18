import numpy as np
import pytest
from pandas.errors import NumbaUtilError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('window,window_kwargs', [['rolling', {'window': 3, 'min_periods': 0}], ['expanding', {}]])
def test_dont_cache_args(self, window, window_kwargs, nogil, parallel, nopython, method):

    def add(values, x):
        return np.sum(values) + x
    engine_kwargs = {'nopython': nopython, 'nogil': nogil, 'parallel': parallel}
    df = DataFrame({'value': [0, 0, 0]})
    result = getattr(df, window)(method=method, **window_kwargs).apply(add, raw=True, engine='numba', engine_kwargs=engine_kwargs, args=(1,))
    expected = DataFrame({'value': [1.0, 1.0, 1.0]})
    tm.assert_frame_equal(result, expected)
    result = getattr(df, window)(method=method, **window_kwargs).apply(add, raw=True, engine='numba', engine_kwargs=engine_kwargs, args=(2,))
    expected = DataFrame({'value': [2.0, 2.0, 2.0]})
    tm.assert_frame_equal(result, expected)