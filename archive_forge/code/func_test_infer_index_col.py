from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@skip_pyarrow
def test_infer_index_col(all_parsers):
    data = 'A,B,C\nfoo,1,2,3\nbar,4,5,6\nbaz,7,8,9\n'
    parser = all_parsers
    result = parser.read_csv(StringIO(data))
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=['foo', 'bar', 'baz'], columns=['A', 'B', 'C'])
    tm.assert_frame_equal(result, expected)