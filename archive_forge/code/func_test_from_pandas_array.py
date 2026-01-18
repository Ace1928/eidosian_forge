from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('dtype', ['M8[ns]', 'm8[ns]'])
def test_from_pandas_array(dtype):
    data = np.array([1, 2, 3], dtype=dtype)
    arr = NumpyExtensionArray(data)
    cls = {'M8[ns]': DatetimeArray, 'm8[ns]': TimedeltaArray}[dtype]
    depr_msg = f'{cls.__name__}.__init__ is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        result = cls(arr)
        expected = cls(data)
    tm.assert_extension_array_equal(result, expected)
    result = cls._from_sequence(arr, dtype=dtype)
    expected = cls._from_sequence(data, dtype=dtype)
    tm.assert_extension_array_equal(result, expected)
    func = {'M8[ns]': pd.to_datetime, 'm8[ns]': pd.to_timedelta}[dtype]
    result = func(arr).array
    expected = func(data).array
    tm.assert_equal(result, expected)
    idx_cls = {'M8[ns]': DatetimeIndex, 'm8[ns]': TimedeltaIndex}[dtype]
    result = idx_cls(arr)
    expected = idx_cls(data)
    tm.assert_index_equal(result, expected)