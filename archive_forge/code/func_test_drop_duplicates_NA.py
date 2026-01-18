from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_drop_duplicates_NA():
    df = DataFrame({'A': [None, None, 'foo', 'bar', 'foo', 'bar', 'bar', 'foo'], 'B': ['one', 'one', 'two', 'two', 'two', 'two', 'one', 'two'], 'C': [1.0, np.nan, np.nan, np.nan, 1.0, 1.0, 1, 1.0], 'D': range(8)})
    result = df.drop_duplicates('A')
    expected = df.loc[[0, 2, 3]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates('A', keep='last')
    expected = df.loc[[1, 6, 7]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates('A', keep=False)
    expected = df.loc[[]]
    tm.assert_frame_equal(result, expected)
    assert len(result) == 0
    result = df.drop_duplicates(['A', 'B'])
    expected = df.loc[[0, 2, 3, 6]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates(['A', 'B'], keep='last')
    expected = df.loc[[1, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates(['A', 'B'], keep=False)
    expected = df.loc[[6]]
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'bar', 'foo'], 'B': ['one', 'one', 'two', 'two', 'two', 'two', 'one', 'two'], 'C': [1.0, np.nan, np.nan, np.nan, 1.0, 1.0, 1, 1.0], 'D': range(8)})
    result = df.drop_duplicates('C')
    expected = df[:2]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates('C', keep='last')
    expected = df.loc[[3, 7]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates('C', keep=False)
    expected = df.loc[[]]
    tm.assert_frame_equal(result, expected)
    assert len(result) == 0
    result = df.drop_duplicates(['C', 'B'])
    expected = df.loc[[0, 1, 2, 4]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates(['C', 'B'], keep='last')
    expected = df.loc[[1, 3, 6, 7]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates(['C', 'B'], keep=False)
    expected = df.loc[[1]]
    tm.assert_frame_equal(result, expected)