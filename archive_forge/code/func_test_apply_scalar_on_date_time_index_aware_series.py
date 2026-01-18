import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
@pytest.mark.parametrize('by_row, expected', [('compat', Series(np.ones(10), dtype='int64')), (False, 1)])
def test_apply_scalar_on_date_time_index_aware_series(by_row, expected):
    series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10, tz='UTC'))
    result = Series(series.index).apply(lambda x: 1, by_row=by_row)
    tm.assert_equal(result, expected)