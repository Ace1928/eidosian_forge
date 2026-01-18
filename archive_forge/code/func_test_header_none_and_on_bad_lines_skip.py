from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
def test_header_none_and_on_bad_lines_skip(all_parsers):
    parser = all_parsers
    data = 'x,1\ny,2,5\nz,3\n'
    result = parser.read_csv(StringIO(data), names=['a', 'b'], header=None, on_bad_lines='skip')
    expected = DataFrame({'a': ['x', 'z'], 'b': [1, 3]})
    tm.assert_frame_equal(result, expected)