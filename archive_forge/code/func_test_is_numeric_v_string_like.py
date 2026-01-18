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
def test_is_numeric_v_string_like():
    assert not com.is_numeric_v_string_like(np.array([1]), 1)
    assert not com.is_numeric_v_string_like(np.array([1]), np.array([2]))
    assert not com.is_numeric_v_string_like(np.array(['foo']), np.array(['foo']))
    assert com.is_numeric_v_string_like(np.array([1]), 'foo')
    assert com.is_numeric_v_string_like(np.array([1, 2]), np.array(['foo']))
    assert com.is_numeric_v_string_like(np.array(['foo']), np.array([1, 2]))