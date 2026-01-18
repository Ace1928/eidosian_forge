from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_drop_duplicates_for_take_all():
    df = DataFrame({'AAA': ['foo', 'bar', 'baz', 'bar', 'foo', 'bar', 'qux', 'foo'], 'B': ['one', 'one', 'two', 'two', 'two', 'two', 'one', 'two'], 'C': [1, 1, 2, 2, 2, 2, 1, 2], 'D': range(8)})
    result = df.drop_duplicates('AAA')
    expected = df.iloc[[0, 1, 2, 6]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates('AAA', keep='last')
    expected = df.iloc[[2, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates('AAA', keep=False)
    expected = df.iloc[[2, 6]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates(['AAA', 'B'])
    expected = df.iloc[[0, 1, 2, 3, 4, 6]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates(['AAA', 'B'], keep='last')
    expected = df.iloc[[0, 1, 2, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates(['AAA', 'B'], keep=False)
    expected = df.iloc[[0, 1, 2, 6]]
    tm.assert_frame_equal(result, expected)