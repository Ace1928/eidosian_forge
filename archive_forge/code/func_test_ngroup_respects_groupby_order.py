from itertools import product
from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ngroup_respects_groupby_order(self, sort):
    df = DataFrame({'a': np.random.default_rng(2).choice(list('abcdef'), 100)})
    g = df.groupby('a', sort=sort)
    df['group_id'] = -1
    df['group_index'] = -1
    for i, (_, group) in enumerate(g):
        df.loc[group.index, 'group_id'] = i
        for j, ind in enumerate(group.index):
            df.loc[ind, 'group_index'] = j
    tm.assert_series_equal(Series(df['group_id'].values), g.ngroup())
    tm.assert_series_equal(Series(df['group_index'].values), g.cumcount())