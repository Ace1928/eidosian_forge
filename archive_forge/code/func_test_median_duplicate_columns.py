from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_median_duplicate_columns():
    df = DataFrame(np.random.default_rng(2).standard_normal((20, 3)), columns=list('aaa'), index=date_range('2012-01-01', periods=20, freq='s'))
    df2 = df.copy()
    df2.columns = ['a', 'b', 'c']
    expected = df2.resample('5s').median()
    result = df.resample('5s').median()
    expected.columns = result.columns
    tm.assert_frame_equal(result, expected)