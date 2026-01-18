from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@skip_pyarrow
def test_empty_with_mangled_column_pass_dtype_by_names(all_parsers):
    parser = all_parsers
    data = 'one,one'
    result = parser.read_csv(StringIO(data), dtype={'one': 'u1', 'one.1': 'f'})
    expected = DataFrame({'one': np.empty(0, dtype='u1'), 'one.1': np.empty(0, dtype='f')})
    tm.assert_frame_equal(result, expected)