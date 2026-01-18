from datetime import (
import numpy as np
import pytest
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import is_dtype_equal
from pandas import (
def test_infer_dtype_from_int_scalar(any_int_numpy_dtype):
    data = np.dtype(any_int_numpy_dtype).type(12)
    dtype, val = infer_dtype_from_scalar(data)
    assert dtype == type(data)