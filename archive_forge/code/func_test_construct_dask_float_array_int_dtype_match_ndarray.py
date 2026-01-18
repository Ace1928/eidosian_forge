import array
import subprocess
import sys
import numpy as np
import pytest
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_construct_dask_float_array_int_dtype_match_ndarray():
    dd = pytest.importorskip('dask.dataframe')
    arr = np.array([1, 2.5, 3])
    darr = dd.from_array(arr)
    res = Series(darr)
    expected = Series(arr)
    tm.assert_series_equal(res, expected)
    msg = 'Trying to coerce float values to integers'
    with pytest.raises(ValueError, match=msg):
        Series(darr, dtype='i8')
    msg = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
    arr[2] = np.nan
    with pytest.raises(IntCastingNaNError, match=msg):
        Series(darr, dtype='i8')
    with pytest.raises(IntCastingNaNError, match=msg):
        Series(arr, dtype='i8')