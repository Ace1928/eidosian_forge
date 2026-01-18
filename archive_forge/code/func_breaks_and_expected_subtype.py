from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_unsigned_integer_dtype
from pandas.core.dtypes.dtypes import IntervalDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
import pandas.core.common as com
@pytest.fixture(params=[([3, 14, 15, 92, 653], np.int64), (np.arange(10, dtype='int64'), np.int64), (Index(np.arange(-10, 11, dtype=np.int64)), np.int64), (Index(np.arange(10, 31, dtype=np.uint64)), np.uint64), (Index(np.arange(20, 30, 0.5), dtype=np.float64), np.float64), (date_range('20180101', periods=10), '<M8[ns]'), (date_range('20180101', periods=10, tz='US/Eastern'), 'datetime64[ns, US/Eastern]'), (timedelta_range('1 day', periods=10), '<m8[ns]')])
def breaks_and_expected_subtype(self, request):
    return request.param