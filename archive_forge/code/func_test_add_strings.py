import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
@pytest.mark.xfail(reason='GH-28527')
def test_add_strings(dtype):
    arr = pd.array(['a', 'b', 'c', 'd'], dtype=dtype)
    df = pd.DataFrame([['t', 'y', 'v', 'w']], dtype=object)
    assert arr.__add__(df) is NotImplemented
    result = arr + df
    expected = pd.DataFrame([['at', 'by', 'cv', 'dw']]).astype(dtype)
    tm.assert_frame_equal(result, expected)
    result = df + arr
    expected = pd.DataFrame([['ta', 'yb', 'vc', 'wd']]).astype(dtype)
    tm.assert_frame_equal(result, expected)