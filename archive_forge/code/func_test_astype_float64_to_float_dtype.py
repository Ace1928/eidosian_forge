import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_astype_float64_to_float_dtype(self, dtype):
    idx = Index([0, 1, 2], dtype=np.float64)
    result = idx.astype(dtype)
    assert isinstance(result, Index) and result.dtype == dtype