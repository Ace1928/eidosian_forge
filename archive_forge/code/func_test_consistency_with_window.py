from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_consistency_with_window(test_frame):
    df = test_frame
    expected = Index([1, 2, 3], name='A')
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('A').resample('2s').mean()
    assert result.index.nlevels == 2
    tm.assert_index_equal(result.index.levels[0], expected)
    result = df.groupby('A').rolling(20).mean()
    assert result.index.nlevels == 2
    tm.assert_index_equal(result.index.levels[0], expected)