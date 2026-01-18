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
@pytest.mark.parametrize('dtype', ['datetime64[ns, US/Eastern]', 'datetime64[ns, Asia/Tokyo]', 'datetime64[ns, UTC]', 'M8[ns, US/Eastern]', 'M8[ns, Asia/Tokyo]', 'M8[ns, UTC]'])
def test_datetimetz_dtype(self, dtype):
    assert com.pandas_dtype(dtype) == DatetimeTZDtype.construct_from_string(dtype)
    assert com.pandas_dtype(dtype) == dtype