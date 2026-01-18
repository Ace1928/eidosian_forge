from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_int_conversion(all_parsers):
    data = 'A,B\n1.0,1\n2.0,2\n3.0,3\n'
    parser = all_parsers
    result = parser.read_csv(StringIO(data))
    expected = DataFrame([[1.0, 1], [2.0, 2], [3.0, 3]], columns=['A', 'B'])
    tm.assert_frame_equal(result, expected)