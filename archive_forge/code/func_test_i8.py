from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
def test_i8(self):
    arr = date_range('20130101', periods=3).values
    result = algos.isin(arr, [arr[0]])
    expected = np.array([True, False, False])
    tm.assert_numpy_array_equal(result, expected)
    result = algos.isin(arr, arr[0:2])
    expected = np.array([True, True, False])
    tm.assert_numpy_array_equal(result, expected)
    result = algos.isin(arr, set(arr[0:2]))
    expected = np.array([True, True, False])
    tm.assert_numpy_array_equal(result, expected)
    arr = timedelta_range('1 day', periods=3).values
    result = algos.isin(arr, [arr[0]])
    expected = np.array([True, False, False])
    tm.assert_numpy_array_equal(result, expected)
    result = algos.isin(arr, arr[0:2])
    expected = np.array([True, True, False])
    tm.assert_numpy_array_equal(result, expected)
    result = algos.isin(arr, set(arr[0:2]))
    expected = np.array([True, True, False])
    tm.assert_numpy_array_equal(result, expected)