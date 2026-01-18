from collections import deque
from datetime import (
from enum import Enum
import functools
import operator
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
def test_inplace_ops_identity(self):
    s_orig = Series([1, 2, 3])
    df_orig = DataFrame(np.random.default_rng(2).integers(0, 5, size=10).reshape(-1, 5))
    s = s_orig.copy()
    s2 = s
    s += 1
    tm.assert_series_equal(s, s2)
    tm.assert_series_equal(s_orig + 1, s)
    assert s is s2
    assert s._mgr is s2._mgr
    df = df_orig.copy()
    df2 = df
    df += 1
    tm.assert_frame_equal(df, df2)
    tm.assert_frame_equal(df_orig + 1, df)
    assert df is df2
    assert df._mgr is df2._mgr
    s = s_orig.copy()
    s2 = s
    s += 1.5
    tm.assert_series_equal(s, s2)
    tm.assert_series_equal(s_orig + 1.5, s)
    df = df_orig.copy()
    df2 = df
    df += 1.5
    tm.assert_frame_equal(df, df2)
    tm.assert_frame_equal(df_orig + 1.5, df)
    assert df is df2
    assert df._mgr is df2._mgr
    arr = np.random.default_rng(2).integers(0, 10, size=5)
    df_orig = DataFrame({'A': arr.copy(), 'B': 'foo'})
    df = df_orig.copy()
    df2 = df
    df['A'] += 1
    expected = DataFrame({'A': arr.copy() + 1, 'B': 'foo'})
    tm.assert_frame_equal(df, expected)
    tm.assert_frame_equal(df2, expected)
    assert df._mgr is df2._mgr
    df = df_orig.copy()
    df2 = df
    df['A'] += 1.5
    expected = DataFrame({'A': arr.copy() + 1.5, 'B': 'foo'})
    tm.assert_frame_equal(df, expected)
    tm.assert_frame_equal(df2, expected)
    assert df._mgr is df2._mgr