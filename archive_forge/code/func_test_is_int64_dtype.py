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
@pytest.mark.parametrize('dtype', [np.int64, np.array([1, 2], dtype=np.int64), 'Int64', pd.Int64Dtype])
def test_is_int64_dtype(dtype):
    msg = 'is_int64_dtype is deprecated'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert com.is_int64_dtype(dtype)