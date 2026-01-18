from __future__ import annotations
from datetime import datetime
import gc
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BaseMaskedArray
def test_ensure_copied_data(self, index):
    init_kwargs = {}
    if isinstance(index, PeriodIndex):
        init_kwargs['freq'] = index.freq
    elif isinstance(index, (RangeIndex, MultiIndex, CategoricalIndex)):
        return
    elif index.dtype == object and index.inferred_type == 'boolean':
        init_kwargs['dtype'] = index.dtype
    index_type = type(index)
    result = index_type(index.values, copy=True, **init_kwargs)
    if is_datetime64tz_dtype(index.dtype):
        result = result.tz_localize('UTC').tz_convert(index.tz)
    if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
        index = index._with_freq(None)
    tm.assert_index_equal(index, result)
    if isinstance(index, PeriodIndex):
        result = index_type(ordinal=index.asi8, copy=False, **init_kwargs)
        tm.assert_numpy_array_equal(index.asi8, result.asi8, check_same='same')
    elif isinstance(index, IntervalIndex):
        pass
    elif type(index) is Index and (not isinstance(index.dtype, np.dtype)):
        result = index_type(index.values, copy=False, **init_kwargs)
        tm.assert_index_equal(result, index)
        if isinstance(index._values, BaseMaskedArray):
            assert np.shares_memory(index._values._data, result._values._data)
            tm.assert_numpy_array_equal(index._values._data, result._values._data, check_same='same')
            assert np.shares_memory(index._values._mask, result._values._mask)
            tm.assert_numpy_array_equal(index._values._mask, result._values._mask, check_same='same')
        elif index.dtype == 'string[python]':
            assert np.shares_memory(index._values._ndarray, result._values._ndarray)
            tm.assert_numpy_array_equal(index._values._ndarray, result._values._ndarray, check_same='same')
        elif index.dtype == 'string[pyarrow]':
            assert tm.shares_memory(result._values, index._values)
        else:
            raise NotImplementedError(index.dtype)
    else:
        result = index_type(index.values, copy=False, **init_kwargs)
        tm.assert_numpy_array_equal(index.values, result.values, check_same='same')