import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_multiindex_assignment_single_dtype(self, using_copy_on_write, warn_copy_on_write):
    arr = np.array([0.0, 1.0])
    df = DataFrame(np.random.default_rng(2).integers(5, 10, size=9).reshape(3, 3), columns=list('abc'), index=[[4, 4, 8], [8, 10, 12]], dtype=np.int64)
    view = df['c'].iloc[:2].values
    df.loc[4, 'c'] = arr
    exp = Series(arr, index=[8, 10], name='c', dtype='int64')
    result = df.loc[4, 'c']
    tm.assert_series_equal(result, exp)
    if not using_copy_on_write:
        tm.assert_numpy_array_equal(view, exp.values)
    with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
        df.loc[4, 'c'] = arr + 0.5
    result = df.loc[4, 'c']
    exp = exp + 0.5
    tm.assert_series_equal(result, exp)
    with tm.assert_cow_warning(warn_copy_on_write):
        df.loc[4, 'c'] = 10
    exp = Series(10, index=[8, 10], name='c', dtype='float64')
    tm.assert_series_equal(df.loc[4, 'c'], exp)
    msg = 'Must have equal len keys and value when setting with an iterable'
    with pytest.raises(ValueError, match=msg):
        df.loc[4, 'c'] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match=msg):
        df.loc[4, 'c'] = [0]
    with tm.assert_cow_warning(warn_copy_on_write):
        df.loc[4, ['c']] = [0]
    assert (df.loc[4, 'c'] == 0).all()