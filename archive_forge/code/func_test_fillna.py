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
def test_fillna(self, index):
    if len(index) == 0:
        return
    elif index.dtype == bool:
        return
    elif isinstance(index, Index) and is_integer_dtype(index.dtype):
        return
    elif isinstance(index, MultiIndex):
        idx = index.copy(deep=True)
        msg = 'isna is not defined for MultiIndex'
        with pytest.raises(NotImplementedError, match=msg):
            idx.fillna(idx[0])
    else:
        idx = index.copy(deep=True)
        result = idx.fillna(idx[0])
        tm.assert_index_equal(result, idx)
        assert result is not idx
        msg = "'value' must be a scalar, passed: "
        with pytest.raises(TypeError, match=msg):
            idx.fillna([idx[0]])
        idx = index.copy(deep=True)
        values = idx._values
        values[1] = np.nan
        idx = type(index)(values)
        msg = "does not support 'downcast'"
        with pytest.raises(NotImplementedError, match=msg):
            idx.fillna(idx[0], downcast='infer')
        expected = np.array([False] * len(idx), dtype=bool)
        expected[1] = True
        tm.assert_numpy_array_equal(idx._isnan, expected)
        assert idx.hasnans is True