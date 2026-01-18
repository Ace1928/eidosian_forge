import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_multiindex(self):
    index1 = MultiIndex.from_arrays([['a', 'a', 'a', 'b', 'b', 'b'], [1, 2, 3, 1, 2, 3]], names=['first', 'second'])
    index2 = MultiIndex.from_arrays([['b', 'b', 'b', 'c', 'c', 'c'], [1, 2, 3, 1, 2, 3]], names=['first', 'second'])
    df1 = DataFrame(data=np.random.default_rng(2).standard_normal(6), index=index1, columns=['var X'])
    df2 = DataFrame(data=np.random.default_rng(2).standard_normal(6), index=index2, columns=['var Y'])
    df1 = df1.sort_index(level=0)
    df2 = df2.sort_index(level=0)
    joined = df1.join(df2, how='outer')
    ex_index = Index(index1.values).union(Index(index2.values))
    expected = df1.reindex(ex_index).join(df2.reindex(ex_index))
    expected.index.names = index1.names
    tm.assert_frame_equal(joined, expected)
    assert joined.index.names == index1.names
    df1 = df1.sort_index(level=1)
    df2 = df2.sort_index(level=1)
    joined = df1.join(df2, how='outer').sort_index(level=0)
    ex_index = Index(index1.values).union(Index(index2.values))
    expected = df1.reindex(ex_index).join(df2.reindex(ex_index))
    expected.index.names = index1.names
    tm.assert_frame_equal(joined, expected)
    assert joined.index.names == index1.names