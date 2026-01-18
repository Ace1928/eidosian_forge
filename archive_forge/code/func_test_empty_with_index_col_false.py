from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@skip_pyarrow
def test_empty_with_index_col_false(all_parsers):
    data = 'x,y'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=False)
    expected = DataFrame(columns=['x', 'y'])
    tm.assert_frame_equal(result, expected)