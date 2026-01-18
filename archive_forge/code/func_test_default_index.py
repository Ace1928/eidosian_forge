from copy import deepcopy
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_default_index(self):
    s1 = Series([1, 2, 3], name='x')
    s2 = Series([4, 5, 6], name='y')
    res = concat([s1, s2], axis=1, ignore_index=True)
    assert isinstance(res.columns, pd.RangeIndex)
    exp = DataFrame([[1, 4], [2, 5], [3, 6]])
    tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)
    s1 = Series([1, 2, 3])
    s2 = Series([4, 5, 6])
    res = concat([s1, s2], axis=1, ignore_index=False)
    assert isinstance(res.columns, pd.RangeIndex)
    exp = DataFrame([[1, 4], [2, 5], [3, 6]])
    exp.columns = pd.RangeIndex(2)
    tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)
    df1 = DataFrame({'A': [1, 2], 'B': [5, 6]})
    df2 = DataFrame({'A': [3, 4], 'B': [7, 8]})
    res = concat([df1, df2], axis=0, ignore_index=True)
    exp = DataFrame([[1, 5], [2, 6], [3, 7], [4, 8]], columns=['A', 'B'])
    tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)
    res = concat([df1, df2], axis=1, ignore_index=True)
    exp = DataFrame([[1, 5, 3, 7], [2, 6, 4, 8]])
    tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)