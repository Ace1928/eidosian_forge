import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_getitem_partial_column_select(self):
    idx = MultiIndex(codes=[[0, 0, 0], [0, 1, 1], [1, 0, 1]], levels=[['a', 'b'], ['x', 'y'], ['p', 'q']])
    df = DataFrame(np.random.default_rng(2).random((3, 2)), index=idx)
    result = df.loc[('a', 'y'), :]
    expected = df.loc['a', 'y']
    tm.assert_frame_equal(result, expected)
    result = df.loc[('a', 'y'), [1, 0]]
    expected = df.loc['a', 'y'][[1, 0]]
    tm.assert_frame_equal(result, expected)
    with pytest.raises(KeyError, match="\\('a', 'foo'\\)"):
        df.loc[('a', 'foo'), :]