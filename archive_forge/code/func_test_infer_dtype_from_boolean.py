from datetime import (
import numpy as np
import pytest
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import is_dtype_equal
from pandas import (
@pytest.mark.parametrize('bool_val', [True, False])
def test_infer_dtype_from_boolean(bool_val):
    dtype, val = infer_dtype_from_scalar(bool_val)
    assert dtype == np.bool_