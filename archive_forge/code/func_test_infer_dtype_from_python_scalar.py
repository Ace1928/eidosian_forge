from datetime import (
import numpy as np
import pytest
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import is_dtype_equal
from pandas import (
@pytest.mark.parametrize('data,exp_dtype', [(12, np.int64), (np.float64(12), np.float64)])
def test_infer_dtype_from_python_scalar(data, exp_dtype):
    dtype, val = infer_dtype_from_scalar(data)
    assert dtype == exp_dtype