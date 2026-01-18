from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_header_with_index_col(all_parsers):
    parser = all_parsers
    data = 'foo,1,2,3\nbar,4,5,6\nbaz,7,8,9\n'
    names = ['A', 'B', 'C']
    result = parser.read_csv(StringIO(data), names=names)
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=['foo', 'bar', 'baz'], columns=['A', 'B', 'C'])
    tm.assert_frame_equal(result, expected)