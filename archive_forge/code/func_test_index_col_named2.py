from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_index_col_named2(all_parsers):
    parser = all_parsers
    data = '1,2,3,4,hello\n5,6,7,8,world\n9,10,11,12,foo\n'
    expected = DataFrame({'a': [1, 5, 9], 'b': [2, 6, 10], 'c': [3, 7, 11], 'd': [4, 8, 12]}, index=Index(['hello', 'world', 'foo'], name='message'))
    names = ['a', 'b', 'c', 'd', 'message']
    result = parser.read_csv(StringIO(data), names=names, index_col=['message'])
    tm.assert_frame_equal(result, expected)