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
def test_different_nans_as_float64(self):
    NAN1 = struct.unpack('d', struct.pack('=Q', 9221120237041090560))[0]
    NAN2 = struct.unpack('d', struct.pack('=Q', 9221120237041090561))[0]
    assert NAN1 != NAN1
    assert NAN2 != NAN2
    arr = np.array([NAN1, NAN2], dtype=np.float64)
    lookup1 = np.array([NAN1], dtype=np.float64)
    result = algos.isin(arr, lookup1)
    expected = np.array([True, True])
    tm.assert_numpy_array_equal(result, expected)
    lookup2 = np.array([NAN2], dtype=np.float64)
    result = algos.isin(arr, lookup2)
    expected = np.array([True, True])
    tm.assert_numpy_array_equal(result, expected)