from datetime import datetime
from io import StringIO
import os
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index_col', [[0, 1], [1, 0]])
def test_multi_index_no_level_names(all_parsers, index_col):
    data = 'index1,index2,A,B,C,D\nfoo,one,2,3,4,5\nfoo,two,7,8,9,10\nfoo,three,12,13,14,15\nbar,one,12,13,14,15\nbar,two,12,13,14,15\n'
    headless_data = '\n'.join(data.split('\n')[1:])
    names = ['A', 'B', 'C', 'D']
    parser = all_parsers
    result = parser.read_csv(StringIO(headless_data), index_col=index_col, header=None, names=names)
    expected = parser.read_csv(StringIO(data), index_col=index_col)
    expected.index.names = [None] * 2
    tm.assert_frame_equal(result, expected)