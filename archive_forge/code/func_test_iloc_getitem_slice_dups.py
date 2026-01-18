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
def test_iloc_getitem_slice_dups(self):
    df1 = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=['A', 'A', 'B', 'B'])
    df2 = DataFrame(np.random.default_rng(2).integers(0, 10, size=20).reshape(10, 2), columns=['A', 'C'])
    df = concat([df1, df2], axis=1)
    tm.assert_frame_equal(df.iloc[:, :4], df1)
    tm.assert_frame_equal(df.iloc[:, 4:], df2)
    df = concat([df2, df1], axis=1)
    tm.assert_frame_equal(df.iloc[:, :2], df2)
    tm.assert_frame_equal(df.iloc[:, 2:], df1)
    exp = concat([df2, df1.iloc[:, [0]]], axis=1)
    tm.assert_frame_equal(df.iloc[:, 0:3], exp)
    df = concat([df, df], axis=0)
    tm.assert_frame_equal(df.iloc[0:10, :2], df2)
    tm.assert_frame_equal(df.iloc[0:10, 2:], df1)
    tm.assert_frame_equal(df.iloc[10:, :2], df2)
    tm.assert_frame_equal(df.iloc[10:, 2:], df1)