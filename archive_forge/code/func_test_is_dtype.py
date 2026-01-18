import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_is_dtype(self, dtype):
    assert IntervalDtype.is_dtype(dtype)
    assert IntervalDtype.is_dtype('interval')
    assert IntervalDtype.is_dtype(IntervalDtype('float64'))
    assert IntervalDtype.is_dtype(IntervalDtype('int64'))
    assert IntervalDtype.is_dtype(IntervalDtype(np.int64))
    assert IntervalDtype.is_dtype(IntervalDtype('float64', 'left'))
    assert IntervalDtype.is_dtype(IntervalDtype('int64', 'right'))
    assert IntervalDtype.is_dtype(IntervalDtype(np.int64, 'both'))
    assert not IntervalDtype.is_dtype('D')
    assert not IntervalDtype.is_dtype('3D')
    assert not IntervalDtype.is_dtype('us')
    assert not IntervalDtype.is_dtype('S')
    assert not IntervalDtype.is_dtype('foo')
    assert not IntervalDtype.is_dtype('IntervalA')
    assert not IntervalDtype.is_dtype(np.object_)
    assert not IntervalDtype.is_dtype(np.int64)
    assert not IntervalDtype.is_dtype(np.float64)