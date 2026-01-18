import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_groupby_transform_with_int():
    df = DataFrame({'A': [1, 1, 1, 2, 2, 2], 'B': Series(1, dtype='float64'), 'C': Series([1, 2, 3, 1, 2, 3], dtype='float64'), 'D': 'foo'})
    with np.errstate(all='ignore'):
        result = df.groupby('A')[['B', 'C']].transform(lambda x: (x - x.mean()) / x.std())
    expected = DataFrame({'B': np.nan, 'C': Series([-1, 0, 1, -1, 0, 1], dtype='float64')})
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'A': [1, 1, 1, 2, 2, 2], 'B': 1, 'C': [1, 2, 3, 1, 2, 3], 'D': 'foo'})
    with np.errstate(all='ignore'):
        with pytest.raises(TypeError, match='Could not convert'):
            df.groupby('A').transform(lambda x: (x - x.mean()) / x.std())
        result = df.groupby('A')[['B', 'C']].transform(lambda x: (x - x.mean()) / x.std())
    expected = DataFrame({'B': np.nan, 'C': [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]})
    tm.assert_frame_equal(result, expected)
    s = Series([2, 3, 4, 10, 5, -1])
    df = DataFrame({'A': [1, 1, 1, 2, 2, 2], 'B': 1, 'C': s, 'D': 'foo'})
    with np.errstate(all='ignore'):
        with pytest.raises(TypeError, match='Could not convert'):
            df.groupby('A').transform(lambda x: (x - x.mean()) / x.std())
        result = df.groupby('A')[['B', 'C']].transform(lambda x: (x - x.mean()) / x.std())
    s1 = s.iloc[0:3]
    s1 = (s1 - s1.mean()) / s1.std()
    s2 = s.iloc[3:6]
    s2 = (s2 - s2.mean()) / s2.std()
    expected = DataFrame({'B': np.nan, 'C': concat([s1, s2])})
    tm.assert_frame_equal(result, expected)
    result = df.groupby('A')[['B', 'C']].transform(lambda x: x * 2 / 2)
    expected = DataFrame({'B': 1.0, 'C': [2.0, 3.0, 4.0, 10.0, 5.0, -1.0]})
    tm.assert_frame_equal(result, expected)