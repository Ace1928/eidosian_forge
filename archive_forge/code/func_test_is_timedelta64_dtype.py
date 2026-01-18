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
def test_is_timedelta64_dtype():
    assert not com.is_timedelta64_dtype(object)
    assert not com.is_timedelta64_dtype(None)
    assert not com.is_timedelta64_dtype([1, 2, 3])
    assert not com.is_timedelta64_dtype(np.array([], dtype=np.datetime64))
    assert not com.is_timedelta64_dtype('0 days')
    assert not com.is_timedelta64_dtype('0 days 00:00:00')
    assert not com.is_timedelta64_dtype(['0 days 00:00:00'])
    assert not com.is_timedelta64_dtype('NO DATE')
    assert com.is_timedelta64_dtype(np.timedelta64)
    assert com.is_timedelta64_dtype(pd.Series([], dtype='timedelta64[ns]'))
    assert com.is_timedelta64_dtype(pd.to_timedelta(['0 days', '1 days']))