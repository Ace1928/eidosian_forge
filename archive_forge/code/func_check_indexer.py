from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError
from pandas.core.dtypes.common import is_list_like
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def check_indexer(self, obj, key, expected, val, indexer, is_inplace):
    orig = obj
    obj = obj.copy()
    arr = obj._values
    indexer(obj)[key] = val
    tm.assert_series_equal(obj, expected)
    self._check_inplace(is_inplace, orig, arr, obj)