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
def test_is_datetime64_any_dtype():
    assert not com.is_datetime64_any_dtype(int)
    assert not com.is_datetime64_any_dtype(str)
    assert not com.is_datetime64_any_dtype(np.array([1, 2]))
    assert not com.is_datetime64_any_dtype(np.array(['a', 'b']))
    assert com.is_datetime64_any_dtype(np.datetime64)
    assert com.is_datetime64_any_dtype(np.array([], dtype=np.datetime64))
    assert com.is_datetime64_any_dtype(DatetimeTZDtype('ns', 'US/Eastern'))
    assert com.is_datetime64_any_dtype(pd.DatetimeIndex([1, 2, 3], dtype='datetime64[ns]'))