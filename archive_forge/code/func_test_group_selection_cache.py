import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_group_selection_cache():
    df = DataFrame([[1, 2], [1, 4], [5, 6]], columns=['A', 'B'])
    expected = df.iloc[[0, 2]]
    g = df.groupby('A')
    result1 = g.head(n=2)
    result2 = g.nth(0)
    tm.assert_frame_equal(result1, df)
    tm.assert_frame_equal(result2, expected)
    g = df.groupby('A')
    result1 = g.tail(n=2)
    result2 = g.nth(0)
    tm.assert_frame_equal(result1, df)
    tm.assert_frame_equal(result2, expected)
    g = df.groupby('A')
    result1 = g.nth(0)
    result2 = g.head(n=2)
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, df)
    g = df.groupby('A')
    result1 = g.nth(0)
    result2 = g.tail(n=2)
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, df)