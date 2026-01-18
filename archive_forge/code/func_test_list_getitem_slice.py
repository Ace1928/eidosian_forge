import re
import pytest
from pandas import (
import pandas._testing as tm
from pandas.compat import pa_version_under11p0
def test_list_getitem_slice():
    ser = Series([[1, 2, 3], [4, None, 5], None], dtype=ArrowDtype(pa.list_(pa.int64())))
    if pa_version_under11p0:
        with pytest.raises(NotImplementedError, match='List slice not supported by pyarrow '):
            ser.list[1:None:None]
    else:
        actual = ser.list[1:None:None]
        expected = Series([[2, 3], [None, 5], None], dtype=ArrowDtype(pa.list_(pa.int64())))
        tm.assert_series_equal(actual, expected)