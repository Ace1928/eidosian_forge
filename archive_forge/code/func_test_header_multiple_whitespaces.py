from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_header_multiple_whitespaces(all_parsers):
    parser = all_parsers
    data = 'aa    bb(1,1)   cc(1,1)\n                0  2  3.5'
    result = parser.read_csv(StringIO(data), sep='\\s+')
    expected = DataFrame({'aa': [0], 'bb(1,1)': 2, 'cc(1,1)': 3.5})
    tm.assert_frame_equal(result, expected)