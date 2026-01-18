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
@pytest.mark.parametrize('from_type', [np.datetime64, np.timedelta64])
def test_astype_object_preserves_datetime_na(from_type):
    arr = np.array([from_type('NaT', 'ns')])
    result = astype_array(arr, dtype=np.dtype('object'))
    assert isna(result)[0]