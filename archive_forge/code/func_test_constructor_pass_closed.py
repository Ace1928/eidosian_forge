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
@pytest.mark.parametrize('breaks', [Index([0, 1, 2, 3, 4], dtype=np.int64), Index([0, 1, 2, 3, 4], dtype=np.uint64), Index([0, 1, 2, 3, 4], dtype=np.float64), date_range('2017-01-01', periods=5), timedelta_range('1 day', periods=5)])
def test_constructor_pass_closed(self, constructor, breaks):
    iv_dtype = IntervalDtype(breaks.dtype)
    result_kwargs = self.get_kwargs_from_breaks(breaks)
    for dtype in (iv_dtype, str(iv_dtype)):
        with tm.assert_produces_warning(None):
            result = constructor(dtype=dtype, closed='left', **result_kwargs)
        assert result.dtype.closed == 'left'