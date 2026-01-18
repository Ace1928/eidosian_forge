from datetime import (
import numpy as np
import pytest
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import is_dtype_equal
from pandas import (
@pytest.mark.parametrize('left, right, subtype', [(0, 1, 'int64'), (0.0, 1.0, 'float64'), (Timestamp(0), Timestamp(1), 'datetime64[ns]'), (Timestamp(0, tz='UTC'), Timestamp(1, tz='UTC'), 'datetime64[ns, UTC]'), (Timedelta(0), Timedelta(1), 'timedelta64[ns]')])
def test_infer_from_interval(left, right, subtype, closed):
    interval = Interval(left, right, closed)
    result_dtype, result_value = infer_dtype_from_scalar(interval)
    expected_dtype = f'interval[{subtype}, {closed}]'
    assert result_dtype == expected_dtype
    assert result_value == interval