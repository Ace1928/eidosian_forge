import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_fillna_nones_inplace():
    df = DataFrame([[None, None], [None, None]], columns=['A', 'B'])
    msg = 'Downcasting object dtype arrays'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.fillna(value={'A': 1, 'B': 2}, inplace=True)
    expected = DataFrame([[1, 2], [1, 2]], columns=['A', 'B'])
    tm.assert_frame_equal(df, expected)