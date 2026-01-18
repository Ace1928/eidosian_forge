from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@skip_pyarrow
def test_header_none_and_implicit_index(all_parsers):
    parser = all_parsers
    data = 'x,1,5\ny,2\nz,3\n'
    result = parser.read_csv(StringIO(data), names=['a', 'b'], header=None)
    expected = DataFrame({'a': [1, 2, 3], 'b': [5, np.nan, np.nan]}, index=['x', 'y', 'z'])
    tm.assert_frame_equal(result, expected)