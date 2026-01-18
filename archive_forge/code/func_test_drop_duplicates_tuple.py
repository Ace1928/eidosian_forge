from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_drop_duplicates_tuple():
    df = DataFrame({('AA', 'AB'): ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'bar', 'foo'], 'B': ['one', 'one', 'two', 'two', 'two', 'two', 'one', 'two'], 'C': [1, 1, 2, 2, 2, 2, 1, 2], 'D': range(8)})
    result = df.drop_duplicates(('AA', 'AB'))
    expected = df[:2]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates(('AA', 'AB'), keep='last')
    expected = df.loc[[6, 7]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates(('AA', 'AB'), keep=False)
    expected = df.loc[[]]
    assert len(result) == 0
    tm.assert_frame_equal(result, expected)
    expected = df.loc[[0, 1, 2, 3]]
    result = df.drop_duplicates((('AA', 'AB'), 'B'))
    tm.assert_frame_equal(result, expected)