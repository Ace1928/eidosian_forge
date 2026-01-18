import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
@pytest.mark.xfail(reason='GH-28527')
def test_add_frame(dtype):
    arr = pd.array(['a', 'b', np.nan, np.nan], dtype=dtype)
    df = pd.DataFrame([['x', np.nan, 'y', np.nan]])
    assert arr.__add__(df) is NotImplemented
    result = arr + df
    expected = pd.DataFrame([['ax', np.nan, np.nan, np.nan]]).astype(dtype)
    tm.assert_frame_equal(result, expected)
    result = df + arr
    expected = pd.DataFrame([['xa', np.nan, np.nan, np.nan]]).astype(dtype)
    tm.assert_frame_equal(result, expected)