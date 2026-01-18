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
@pytest.mark.parametrize('dtype', ['int8', 'int16'])
def test_diff_low_precision_int(self, dtype):
    arr = np.array([0, 1, 1, 0, 0], dtype=dtype)
    result = algos.diff(arr, 1)
    expected = np.array([np.nan, 1, 0, -1, 0], dtype='float32')
    tm.assert_numpy_array_equal(result, expected)