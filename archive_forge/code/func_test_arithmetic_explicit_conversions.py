from __future__ import annotations
from datetime import datetime
import gc
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BaseMaskedArray
def test_arithmetic_explicit_conversions(self):
    index_cls = self._index_cls
    if index_cls is RangeIndex:
        idx = RangeIndex(5)
    else:
        idx = index_cls(np.arange(5, dtype='int64'))
    arr = np.arange(5, dtype='int64') * 3.2
    expected = Index(arr, dtype=np.float64)
    fidx = idx * 3.2
    tm.assert_index_equal(fidx, expected)
    fidx = 3.2 * idx
    tm.assert_index_equal(fidx, expected)
    expected = Index(arr, dtype=np.float64)
    a = np.zeros(5, dtype='float64')
    result = fidx - a
    tm.assert_index_equal(result, expected)
    expected = Index(-arr, dtype=np.float64)
    a = np.zeros(5, dtype='float64')
    result = a - fidx
    tm.assert_index_equal(result, expected)