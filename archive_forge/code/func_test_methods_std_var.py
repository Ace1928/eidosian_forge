from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
@pytest.mark.parametrize('f', ['std', 'var'])
def test_methods_std_var(f, test_frame):
    g = test_frame.groupby('A')
    r = g.resample('2s')
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = getattr(r, f)(ddof=1)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        expected = g.apply(lambda x: getattr(x.resample('2s'), f)(ddof=1))
    tm.assert_frame_equal(result, expected)