from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@skip_pyarrow
def test_infer_types_boolean_sum(all_parsers):
    parser = all_parsers
    result = parser.read_csv(StringIO('0,1'), names=['a', 'b'], index_col=['a'], dtype={'a': 'UInt8'})
    expected = DataFrame(data={'a': [0], 'b': [1]}).set_index('a')
    tm.assert_frame_equal(result, expected, check_index_type=False)