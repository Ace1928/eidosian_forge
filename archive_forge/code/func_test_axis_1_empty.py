from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
@pytest.mark.parametrize('index', [RangeIndex(0), DatetimeIndex([]), Index([], dtype=np.int64), Index([], dtype=np.float64), DatetimeIndex([], freq='ME'), PeriodIndex([], freq='D')])
def test_axis_1_empty(self, all_reductions, index):
    df = DataFrame(columns=['a'], index=index)
    result = getattr(df, all_reductions)(axis=1)
    if all_reductions in ('any', 'all'):
        expected_dtype = 'bool'
    elif all_reductions == 'count':
        expected_dtype = 'int64'
    else:
        expected_dtype = 'object'
    expected = Series([], index=index, dtype=expected_dtype)
    tm.assert_series_equal(result, expected)