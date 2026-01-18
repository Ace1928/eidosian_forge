import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_empty_series(self):
    s1 = Series([1, 2, 3], name='x')
    s2 = Series(name='y', dtype='float64')
    res = concat([s1, s2], axis=1)
    exp = DataFrame({'x': [1, 2, 3], 'y': [np.nan, np.nan, np.nan]}, index=RangeIndex(3))
    tm.assert_frame_equal(res, exp)
    s1 = Series([1, 2, 3], name='x')
    s2 = Series(name='y', dtype='float64')
    msg = 'The behavior of array concatenation with empty entries is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = concat([s1, s2], axis=0)
    exp = Series([1, 2, 3])
    tm.assert_series_equal(res, exp)
    s1 = Series([1, 2, 3], name='x')
    s2 = Series(name=None, dtype='float64')
    res = concat([s1, s2], axis=1)
    exp = DataFrame({'x': [1, 2, 3], 0: [np.nan, np.nan, np.nan]}, columns=['x', 0], index=RangeIndex(3))
    tm.assert_frame_equal(res, exp)