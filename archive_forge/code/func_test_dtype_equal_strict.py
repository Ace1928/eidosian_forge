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
@pytest.mark.parametrize('dtype1,dtype2', [(np.int8, np.int64), (np.int16, np.int64), (np.int32, np.int64), (np.float32, np.float64), (PeriodDtype('D'), PeriodDtype('2D')), (com.pandas_dtype('datetime64[ns, US/Eastern]'), com.pandas_dtype('datetime64[ns, CET]')), (None, None)])
def test_dtype_equal_strict(dtype1, dtype2):
    assert not com.is_dtype_equal(dtype1, dtype2)