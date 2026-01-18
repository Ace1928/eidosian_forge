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
def test_mask_key(self, obj, key, expected, warn, val, indexer_sli):
    mask = np.zeros(obj.shape, dtype=bool)
    mask[key] = True
    obj = obj.copy()
    if is_list_like(val) and len(val) < mask.sum():
        msg = 'boolean index did not match indexed array along dimension'
        with pytest.raises(IndexError, match=msg):
            indexer_sli(obj)[mask] = val
        return
    with tm.assert_produces_warning(warn, match='incompatible dtype'):
        indexer_sli(obj)[mask] = val
    tm.assert_series_equal(obj, expected)