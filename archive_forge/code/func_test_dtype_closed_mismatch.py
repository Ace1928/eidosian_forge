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
def test_dtype_closed_mismatch():
    dtype = IntervalDtype(np.int64, 'left')
    msg = 'closed keyword does not match dtype.closed'
    with pytest.raises(ValueError, match=msg):
        IntervalIndex([], dtype=dtype, closed='neither')
    with pytest.raises(ValueError, match=msg):
        IntervalArray([], dtype=dtype, closed='neither')