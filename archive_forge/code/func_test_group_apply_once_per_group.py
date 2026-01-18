from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('df, group_names', [(DataFrame({'a': [1, 1, 1, 2, 3], 'b': ['a', 'a', 'a', 'b', 'c']}), [1, 2, 3]), (DataFrame({'a': [0, 0, 1, 1], 'b': [0, 1, 0, 1]}), [0, 1]), (DataFrame({'a': [1]}), [1]), (DataFrame({'a': [1, 1, 1, 2, 2, 1, 1, 2], 'b': range(8)}), [1, 2]), (DataFrame({'a': [1, 2, 3, 1, 2, 3], 'two': [4, 5, 6, 7, 8, 9]}), [1, 2, 3]), (DataFrame({'a': list('aaabbbcccc'), 'B': [3, 4, 3, 6, 5, 2, 1, 9, 5, 4], 'C': [4, 0, 2, 2, 2, 7, 8, 6, 2, 8]}), ['a', 'b', 'c']), (DataFrame([[1, 2, 3], [2, 2, 3]], columns=['a', 'b', 'c']), [1, 2])], ids=['GH2936', 'GH7739 & GH10519', 'GH10519', 'GH2656', 'GH12155', 'GH20084', 'GH21417'])
def test_group_apply_once_per_group(df, group_names):
    names = []

    def f_copy(group):
        names.append(group.name)
        return group.copy()

    def f_nocopy(group):
        names.append(group.name)
        return group

    def f_scalar(group):
        names.append(group.name)
        return 0

    def f_none(group):
        names.append(group.name)

    def f_constant_df(group):
        names.append(group.name)
        return DataFrame({'a': [1], 'b': [1]})
    for func in [f_copy, f_nocopy, f_scalar, f_none, f_constant_df]:
        del names[:]
        msg = 'DataFrameGroupBy.apply operated on the grouping columns'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            df.groupby('a', group_keys=False).apply(func)
        assert names == group_names