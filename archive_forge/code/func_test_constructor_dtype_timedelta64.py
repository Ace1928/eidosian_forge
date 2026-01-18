from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
def test_constructor_dtype_timedelta64(self):
    td = Series([timedelta(days=i) for i in range(3)])
    assert td.dtype == 'timedelta64[ns]'
    td = Series([timedelta(days=1)])
    assert td.dtype == 'timedelta64[ns]'
    td = Series([timedelta(days=1), timedelta(days=2), np.timedelta64(1, 's')])
    assert td.dtype == 'timedelta64[ns]'
    td = Series([timedelta(days=1), NaT], dtype='m8[ns]')
    assert td.dtype == 'timedelta64[ns]'
    td = Series([timedelta(days=1), np.nan], dtype='m8[ns]')
    assert td.dtype == 'timedelta64[ns]'
    td = Series([np.timedelta64(300000000), NaT], dtype='m8[ns]')
    assert td.dtype == 'timedelta64[ns]'
    td = Series([np.timedelta64(300000000), NaT])
    assert td.dtype == 'timedelta64[ns]'
    td = Series([np.timedelta64(300000000), iNaT])
    assert td.dtype == 'object'
    td = Series([np.timedelta64(300000000), np.nan])
    assert td.dtype == 'timedelta64[ns]'
    td = Series([NaT, np.timedelta64(300000000)])
    assert td.dtype == 'timedelta64[ns]'
    td = Series([np.timedelta64(1, 's')])
    assert td.dtype == 'timedelta64[ns]'
    td.astype('int64')
    msg = 'Converting from timedelta64\\[ns\\] to int32 is not supported'
    with pytest.raises(TypeError, match=msg):
        td.astype('int32')
    msg = '|'.join(['Could not convert object to NumPy timedelta', "Could not convert 'foo' to NumPy timedelta"])
    with pytest.raises(ValueError, match=msg):
        Series([timedelta(days=1), 'foo'], dtype='m8[ns]')
    td = Series([timedelta(days=i) for i in range(3)] + ['foo'])
    assert td.dtype == 'object'
    ser = Series([None, NaT, '1 Day'])
    assert ser.dtype == object
    ser = Series([np.nan, NaT, '1 Day'])
    assert ser.dtype == object
    ser = Series([NaT, None, '1 Day'])
    assert ser.dtype == object
    ser = Series([NaT, np.nan, '1 Day'])
    assert ser.dtype == object