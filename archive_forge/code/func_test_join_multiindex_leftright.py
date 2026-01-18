from datetime import datetime
import numpy as np
import pytest
from pandas.errors import MergeError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
def test_join_multiindex_leftright(self):
    df1 = DataFrame([['a', 'x', 0.47178], ['a', 'y', 0.774908], ['a', 'z', 0.563634], ['b', 'x', -0.353756], ['b', 'y', 0.368062], ['b', 'z', -1.72184], ['c', 'x', 1], ['c', 'y', 2], ['c', 'z', 3]], columns=['first', 'second', 'value1']).set_index(['first', 'second'])
    df2 = DataFrame([['a', 10], ['b', 20]], columns=['first', 'value2']).set_index(['first'])
    exp = DataFrame([[0.47178, 10], [0.774908, 10], [0.563634, 10], [-0.353756, 20], [0.368062, 20], [-1.72184, 20], [1.0, np.nan], [2.0, np.nan], [3.0, np.nan]], index=df1.index, columns=['value1', 'value2'])
    tm.assert_frame_equal(df1.join(df2, how='left'), exp)
    tm.assert_frame_equal(df2.join(df1, how='right'), exp[['value2', 'value1']])
    exp_idx = MultiIndex.from_product([['a', 'b'], ['x', 'y', 'z']], names=['first', 'second'])
    exp = DataFrame([[0.47178, 10], [0.774908, 10], [0.563634, 10], [-0.353756, 20], [0.368062, 20], [-1.72184, 20]], index=exp_idx, columns=['value1', 'value2'])
    tm.assert_frame_equal(df1.join(df2, how='right'), exp)
    tm.assert_frame_equal(df2.join(df1, how='left'), exp[['value2', 'value1']])