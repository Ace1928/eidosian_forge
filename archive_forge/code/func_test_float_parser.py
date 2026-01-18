from io import StringIO
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas import DataFrame
import pandas._testing as tm
@skip_pyarrow
def test_float_parser(all_parsers):
    parser = all_parsers
    data = '45e-1,4.5,45.,inf,-inf'
    result = parser.read_csv(StringIO(data), header=None)
    expected = DataFrame([[float(s) for s in data.split(',')]])
    tm.assert_frame_equal(result, expected)