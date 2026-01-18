import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_isin_multiIndex(self):
    idx = MultiIndex.from_tuples([(0, 'a', 'foo'), (0, 'a', 'bar'), (0, 'b', 'bar'), (0, 'b', 'baz'), (2, 'a', 'foo'), (2, 'a', 'bar'), (2, 'c', 'bar'), (2, 'c', 'baz'), (1, 'b', 'foo'), (1, 'b', 'bar'), (1, 'c', 'bar'), (1, 'c', 'baz')])
    df1 = DataFrame({'A': np.ones(12), 'B': np.zeros(12)}, index=idx)
    df2 = DataFrame({'A': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 'B': [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1]})
    expected = DataFrame(False, index=df1.index, columns=df1.columns)
    result = df1.isin(df2)
    tm.assert_frame_equal(result, expected)
    df2.index = idx
    expected = df2.values.astype(bool)
    expected[:, 1] = ~expected[:, 1]
    expected = DataFrame(expected, columns=['A', 'B'], index=idx)
    result = df1.isin(df2)
    tm.assert_frame_equal(result, expected)