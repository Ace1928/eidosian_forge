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
def test_index_where(self, obj, key, expected, warn, val, using_infer_string):
    mask = np.zeros(obj.shape, dtype=bool)
    mask[key] = True
    if using_infer_string and obj.dtype == object:
        with pytest.raises(TypeError, match='Scalar must'):
            Index(obj).where(~mask, val)
    else:
        res = Index(obj).where(~mask, val)
        expected_idx = Index(expected, dtype=expected.dtype)
        tm.assert_index_equal(res, expected_idx)