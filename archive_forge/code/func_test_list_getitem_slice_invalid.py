import re
import pytest
from pandas import (
import pandas._testing as tm
from pandas.compat import pa_version_under11p0
def test_list_getitem_slice_invalid():
    ser = Series([[1, 2, 3], [4, None, 5], None], dtype=ArrowDtype(pa.list_(pa.int64())))
    if pa_version_under11p0:
        with pytest.raises(NotImplementedError, match='List slice not supported by pyarrow '):
            ser.list[1:None:0]
    else:
        with pytest.raises(pa.lib.ArrowInvalid, match=re.escape('`step` must be >= 1')):
            ser.list[1:None:0]