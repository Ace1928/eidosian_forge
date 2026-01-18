import re
import pytest
from pandas import (
import pandas._testing as tm
from pandas.compat import pa_version_under11p0
def test_list_flatten():
    ser = Series([[1, 2, 3], [4, None], None], dtype=ArrowDtype(pa.list_(pa.int64())))
    actual = ser.list.flatten()
    expected = Series([1, 2, 3, 4, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(actual, expected)