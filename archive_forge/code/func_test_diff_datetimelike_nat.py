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
@pytest.mark.parametrize('dtype', ['M8[ns]', 'm8[ns]'])
def test_diff_datetimelike_nat(self, dtype):
    arr = np.arange(12).astype(np.int64).view(dtype).reshape(3, 4)
    arr[:, 2] = arr.dtype.type('NaT', 'ns')
    result = algos.diff(arr, 1, axis=0)
    expected = np.ones(arr.shape, dtype='timedelta64[ns]') * 4
    expected[:, 2] = np.timedelta64('NaT', 'ns')
    expected[0, :] = np.timedelta64('NaT', 'ns')
    tm.assert_numpy_array_equal(result, expected)
    result = algos.diff(arr.T, 1, axis=1)
    tm.assert_numpy_array_equal(result, expected.T)