from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_int64_min_issues(all_parsers):
    parser = all_parsers
    data = 'A,B\n0,0\n0,'
    result = parser.read_csv(StringIO(data))
    expected = DataFrame({'A': [0, 0], 'B': [0, np.nan]})
    tm.assert_frame_equal(result, expected)