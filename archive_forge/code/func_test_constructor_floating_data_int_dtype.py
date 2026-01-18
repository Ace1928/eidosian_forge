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
def test_constructor_floating_data_int_dtype(self, frame_or_series):
    arr = np.random.default_rng(2).standard_normal(2)
    msg = 'Trying to coerce float values to integer'
    with pytest.raises(ValueError, match=msg):
        frame_or_series(arr, dtype='i8')
    with pytest.raises(ValueError, match=msg):
        frame_or_series(list(arr), dtype='i8')
    arr[0] = np.nan
    msg = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
    with pytest.raises(IntCastingNaNError, match=msg):
        frame_or_series(arr, dtype='i8')
    exc = IntCastingNaNError
    if frame_or_series is Series:
        exc = ValueError
        msg = 'cannot convert float NaN to integer'
    with pytest.raises(exc, match=msg):
        frame_or_series(list(arr), dtype='i8')
    arr = np.array([1.0, 2.0], dtype='float64')
    expected = frame_or_series(arr.astype('i8'))
    obj = frame_or_series(arr, dtype='i8')
    tm.assert_equal(obj, expected)
    obj = frame_or_series(list(arr), dtype='i8')
    tm.assert_equal(obj, expected)