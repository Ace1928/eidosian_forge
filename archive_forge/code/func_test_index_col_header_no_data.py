from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_index_col_header_no_data(all_parsers):
    parser = all_parsers
    result = parser.read_csv(StringIO('a0,a1,a2\n'), header=[0], index_col=0)
    expected = DataFrame([], columns=['a1', 'a2'], index=Index([], name='a0'))
    tm.assert_frame_equal(result, expected)