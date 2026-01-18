from datetime import (
import numpy as np
import pytest
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import is_dtype_equal
from pandas import (
@pytest.mark.parametrize('cls', [np.datetime64, np.timedelta64])
def test_infer_dtype_from_scalar_zerodim_datetimelike(cls):
    val = cls(1234, 'ns')
    arr = np.array(val)
    dtype, res = infer_dtype_from_scalar(arr)
    assert dtype.type is cls
    assert isinstance(res, cls)
    dtype, res = infer_dtype_from(arr)
    assert dtype.type is cls