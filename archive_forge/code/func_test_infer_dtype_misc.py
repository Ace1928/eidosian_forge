from datetime import (
import numpy as np
import pytest
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import is_dtype_equal
from pandas import (
def test_infer_dtype_misc():
    dt = date(2000, 1, 1)
    dtype, val = infer_dtype_from_scalar(dt)
    assert dtype == np.object_
    ts = Timestamp(1, tz='US/Eastern')
    dtype, val = infer_dtype_from_scalar(ts)
    assert dtype == 'datetime64[ns, US/Eastern]'