from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@skip_pyarrow
def test_empty_with_index_pass_dtype(all_parsers):
    parser = all_parsers
    data = 'one,two'
    result = parser.read_csv(StringIO(data), index_col=['one'], dtype={'one': 'u1', 1: 'f'})
    expected = DataFrame({'two': np.empty(0, dtype='f')}, index=Index([], dtype='u1', name='one'))
    tm.assert_frame_equal(result, expected)