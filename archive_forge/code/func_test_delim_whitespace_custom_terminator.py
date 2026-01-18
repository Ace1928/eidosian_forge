from decimal import Decimal
from io import (
import mmap
import os
import tarfile
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_delim_whitespace_custom_terminator(c_parser_only):
    data = 'a b c~1 2 3~4 5 6~7 8 9'
    parser = c_parser_only
    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
        df = parser.read_csv(StringIO(data), lineterminator='~', delim_whitespace=True)
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['a', 'b', 'c'])
    tm.assert_frame_equal(df, expected)