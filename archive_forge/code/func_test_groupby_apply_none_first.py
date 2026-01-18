from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_groupby_apply_none_first():
    test_df1 = DataFrame({'groups': [1, 1, 1, 2], 'vars': [0, 1, 2, 3]})
    test_df2 = DataFrame({'groups': [1, 2, 2, 2], 'vars': [0, 1, 2, 3]})

    def test_func(x):
        if x.shape[0] < 2:
            return None
        return x.iloc[[0, -1]]
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result1 = test_df1.groupby('groups').apply(test_func)
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result2 = test_df2.groupby('groups').apply(test_func)
    index1 = MultiIndex.from_arrays([[1, 1], [0, 2]], names=['groups', None])
    index2 = MultiIndex.from_arrays([[2, 2], [1, 3]], names=['groups', None])
    expected1 = DataFrame({'groups': [1, 1], 'vars': [0, 2]}, index=index1)
    expected2 = DataFrame({'groups': [2, 2], 'vars': [1, 3]}, index=index2)
    tm.assert_frame_equal(result1, expected1)
    tm.assert_frame_equal(result2, expected2)