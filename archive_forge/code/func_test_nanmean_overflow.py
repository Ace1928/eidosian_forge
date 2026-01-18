from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@pytest.mark.parametrize('val', [2 ** 55, -2 ** 55, 20150515061816532])
def test_nanmean_overflow(disable_bottleneck, val):
    ser = Series(val, index=range(500), dtype=np.int64)
    result = ser.mean()
    np_result = ser.values.mean()
    assert result == val
    assert result == np_result
    assert result.dtype == np.float64