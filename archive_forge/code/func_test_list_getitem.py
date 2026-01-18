import re
import pytest
from pandas import (
import pandas._testing as tm
from pandas.compat import pa_version_under11p0
@pytest.mark.parametrize('list_dtype', (pa.list_(pa.int64()), pa.list_(pa.int64(), list_size=3), pa.large_list(pa.int64())))
def test_list_getitem(list_dtype):
    ser = Series([[1, 2, 3], [4, None, 5], None], dtype=ArrowDtype(list_dtype))
    actual = ser.list[1]
    expected = Series([2, None, None], dtype='int64[pyarrow]')
    tm.assert_series_equal(actual, expected)