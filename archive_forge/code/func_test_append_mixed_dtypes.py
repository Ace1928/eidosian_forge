import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_append_mixed_dtypes():
    dti = date_range('2011-01-01', freq='ME', periods=3)
    dti_tz = date_range('2011-01-01', freq='ME', periods=3, tz='US/Eastern')
    pi = period_range('2011-01', freq='M', periods=3)
    mi = MultiIndex.from_arrays([[1, 2, 3], [1.1, np.nan, 3.3], ['a', 'b', 'c'], dti, dti_tz, pi])
    assert mi.nlevels == 6
    res = mi.append(mi)
    exp = MultiIndex.from_arrays([[1, 2, 3, 1, 2, 3], [1.1, np.nan, 3.3, 1.1, np.nan, 3.3], ['a', 'b', 'c', 'a', 'b', 'c'], dti.append(dti), dti_tz.append(dti_tz), pi.append(pi)])
    tm.assert_index_equal(res, exp)
    other = MultiIndex.from_arrays([['x', 'y', 'z'], ['x', 'y', 'z'], ['x', 'y', 'z'], ['x', 'y', 'z'], ['x', 'y', 'z'], ['x', 'y', 'z']])
    res = mi.append(other)
    exp = MultiIndex.from_arrays([[1, 2, 3, 'x', 'y', 'z'], [1.1, np.nan, 3.3, 'x', 'y', 'z'], ['a', 'b', 'c', 'x', 'y', 'z'], dti.append(Index(['x', 'y', 'z'])), dti_tz.append(Index(['x', 'y', 'z'])), pi.append(Index(['x', 'y', 'z']))])
    tm.assert_index_equal(res, exp)