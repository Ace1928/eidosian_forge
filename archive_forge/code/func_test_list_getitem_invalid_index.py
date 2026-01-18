import re
import pytest
from pandas import (
import pandas._testing as tm
from pandas.compat import pa_version_under11p0
@pytest.mark.parametrize('list_dtype', (pa.list_(pa.int64()), pa.list_(pa.int64(), list_size=3), pa.large_list(pa.int64())))
def test_list_getitem_invalid_index(list_dtype):
    ser = Series([[1, 2, 3], [4, None, 5], None], dtype=ArrowDtype(list_dtype))
    with pytest.raises(pa.lib.ArrowInvalid, match='Index -1 is out of bounds'):
        ser.list[-1]
    with pytest.raises(pa.lib.ArrowInvalid, match='Index 5 is out of bounds'):
        ser.list[5]
    with pytest.raises(ValueError, match='key must be an int or slice, got str'):
        ser.list['abc']