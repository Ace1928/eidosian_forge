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
def test_array_interface(self, arr1d):
    arr = arr1d
    result = np.asarray(arr)
    expected = np.array(list(arr), dtype=object)
    tm.assert_numpy_array_equal(result, expected)
    result = np.asarray(arr, dtype=object)
    tm.assert_numpy_array_equal(result, expected)
    result = np.asarray(arr, dtype='int64')
    tm.assert_numpy_array_equal(result, arr.asi8)
    msg = "float\\(\\) argument must be a string or a( real)? number, not 'Period'"
    with pytest.raises(TypeError, match=msg):
        np.asarray(arr, dtype='float64')
    result = np.asarray(arr, dtype='S20')
    expected = np.asarray(arr).astype('S20')
    tm.assert_numpy_array_equal(result, expected)