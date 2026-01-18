from datetime import datetime
import numpy as np
import pytest
from pytz import UTC
from pandas._libs.tslibs import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('copy', [True, False])
@pytest.mark.parametrize('dtype', ['M8[ns]', 'M8[s]'])
def test_length_zero_copy(dtype, copy):
    arr = np.array([], dtype=dtype)
    result = astype_overflowsafe(arr, copy=copy, dtype=np.dtype('M8[ns]'))
    if copy:
        assert not np.shares_memory(result, arr)
    elif arr.dtype == result.dtype:
        assert result is arr
    else:
        assert not np.shares_memory(result, arr)