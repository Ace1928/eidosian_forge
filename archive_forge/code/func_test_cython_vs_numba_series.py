import pytest
from pandas import (
import pandas._testing as tm
def test_cython_vs_numba_series(self, sort, nogil, parallel, nopython, numba_supported_reductions):
    func, kwargs = numba_supported_reductions
    ser = Series(range(3), index=[1, 2, 1], name='foo')
    engine_kwargs = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
    gb = ser.groupby(level=0, sort=sort)
    result = getattr(gb, func)(engine='numba', engine_kwargs=engine_kwargs, **kwargs)
    expected = getattr(gb, func)(**kwargs)
    tm.assert_series_equal(result, expected)