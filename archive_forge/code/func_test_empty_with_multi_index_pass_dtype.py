from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@skip_pyarrow
def test_empty_with_multi_index_pass_dtype(all_parsers):
    parser = all_parsers
    data = 'one,two,three'
    result = parser.read_csv(StringIO(data), index_col=['one', 'two'], dtype={'one': 'u1', 1: 'f8'})
    exp_idx = MultiIndex.from_arrays([np.empty(0, dtype='u1'), np.empty(0, dtype=np.float64)], names=['one', 'two'])
    expected = DataFrame({'three': np.empty(0, dtype=object)}, index=exp_idx)
    tm.assert_frame_equal(result, expected)