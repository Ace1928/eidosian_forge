from __future__ import annotations
from typing import final
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
def test_add_series_with_extension_array(self, data):
    ser = pd.Series(data)
    exc = self._get_expected_exception('__add__', ser, data)
    if exc is not None:
        with pytest.raises(exc):
            ser + data
        return
    result = ser + data
    expected = pd.Series(data + data)
    tm.assert_series_equal(result, expected)