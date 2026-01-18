from __future__ import annotations
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.astype import astype_array
import pandas.core.dtypes.common as com
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
import pandas as pd
import pandas._testing as tm
from pandas.api.types import pandas_dtype
from pandas.arrays import SparseArray
@pytest.mark.parametrize('dtype, expected', [(int, False), (pd.Series([1, 2]), False), (str, True), (object, True), (np.array(['a', 'b']), True), (pd.StringDtype(), True), (pd.Index([], dtype='O'), True)])
def test_is_string_dtype(dtype, expected):
    result = com.is_string_dtype(dtype)
    assert result is expected