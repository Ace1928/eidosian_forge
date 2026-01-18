import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data', [['1', 2, 3], [1, 2, 3], np.array(['1970-01-02', '1970-01-03', '1970-01-04'], dtype='datetime64[D]')])
@pytest.mark.parametrize('kwargs,exp_dtype', [({}, np.int64), ({'downcast': None}, np.int64), ({'downcast': 'float'}, np.dtype(np.float32).char), ({'downcast': 'unsigned'}, np.dtype(np.typecodes['UnsignedInteger'][0]))])
def test_downcast_basic(data, kwargs, exp_dtype):
    result = to_numeric(data, **kwargs)
    expected = np.array([1, 2, 3], dtype=exp_dtype)
    tm.assert_numpy_array_equal(result, expected)