import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['int16', 'int32', 'int64'])
def test_astype_float64_to_int_dtype(self, dtype):
    idx = Index([0, 1, 2], dtype=np.float64)
    result = idx.astype(dtype)
    expected = Index([0, 1, 2], dtype=dtype)
    tm.assert_index_equal(result, expected, exact=True)
    idx = Index([0, 1.1, 2], dtype=np.float64)
    result = idx.astype(dtype)
    expected = Index([0, 1, 2], dtype=dtype)
    tm.assert_index_equal(result, expected, exact=True)