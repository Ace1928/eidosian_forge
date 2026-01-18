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
def test_iloc_getitem_float_duplicates(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), index=[0.1, 0.2, 0.2], columns=list('abc'))
    expect = df.iloc[1:]
    tm.assert_frame_equal(df.loc[0.2], expect)
    expect = df.iloc[1:, 0]
    tm.assert_series_equal(df.loc[0.2, 'a'], expect)
    df.index = [1, 0.2, 0.2]
    expect = df.iloc[1:]
    tm.assert_frame_equal(df.loc[0.2], expect)
    expect = df.iloc[1:, 0]
    tm.assert_series_equal(df.loc[0.2, 'a'], expect)
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 3)), index=[1, 0.2, 0.2, 1], columns=list('abc'))
    expect = df.iloc[1:-1]
    tm.assert_frame_equal(df.loc[0.2], expect)
    expect = df.iloc[1:-1, 0]
    tm.assert_series_equal(df.loc[0.2, 'a'], expect)
    df.index = [0.1, 0.2, 2, 0.2]
    expect = df.iloc[[1, -1]]
    tm.assert_frame_equal(df.loc[0.2], expect)
    expect = df.iloc[[1, -1], 0]
    tm.assert_series_equal(df.loc[0.2, 'a'], expect)