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
@pytest.mark.parametrize('numeric_only', [True, False, None])
@pytest.mark.parametrize('method', ['min', 'max'])
def test_minmax_extensionarray(method, numeric_only):
    int64_info = np.iinfo('int64')
    ser = Series([int64_info.max, None, int64_info.min], dtype=pd.Int64Dtype())
    df = DataFrame({'Int64': ser})
    result = getattr(df, method)(numeric_only=numeric_only)
    expected = Series([getattr(int64_info, method)], dtype='Int64', index=Index(['Int64']))
    tm.assert_series_equal(result, expected)