from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_iloc_setitem_dups(self):
    df1 = DataFrame([{'A': None, 'B': 1}, {'A': 2, 'B': 2}])
    df2 = DataFrame([{'A': 3, 'B': 3}, {'A': 4, 'B': 4}])
    df = concat([df1, df2], axis=1)
    expected = df.fillna(3)
    inds = np.isnan(df.iloc[:, 0])
    mask = inds[inds].index
    df.iloc[mask, 0] = df.iloc[mask, 2]
    tm.assert_frame_equal(df, expected)
    expected = DataFrame({0: [1, 2], 1: [3, 4]})
    expected.columns = ['B', 'B']
    del df['A']
    tm.assert_frame_equal(df, expected)
    df.iloc[[0, 1], [0, 1]] = df.iloc[[0, 1], [0, 1]]
    tm.assert_frame_equal(df, expected)
    df.iloc[[1, 0], [0, 1]] = df.iloc[[1, 0], [0, 1]].reset_index(drop=True)
    df.iloc[[1, 0], [0, 1]] = df.iloc[[1, 0], [0, 1]].reset_index(drop=True)
    tm.assert_frame_equal(df, expected)