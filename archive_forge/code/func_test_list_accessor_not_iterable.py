import re
import pytest
from pandas import (
import pandas._testing as tm
from pandas.compat import pa_version_under11p0
def test_list_accessor_not_iterable():
    ser = Series([[1, 2, 3], [4, None], None], dtype=ArrowDtype(pa.list_(pa.int64())))
    with pytest.raises(TypeError, match="'ListAccessor' object is not iterable"):
        iter(ser.list)