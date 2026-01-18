import numpy as np
import pytest
from pandas.core.dtypes import dtypes
from pandas.core.dtypes.common import is_extension_array_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
def test_astype():
    arr = DummyArray(np.array([1, 2, 3]))
    expected = np.array([1, 2, 3], dtype=object)
    result = arr.astype(object)
    tm.assert_numpy_array_equal(result, expected)
    result = arr.astype('object')
    tm.assert_numpy_array_equal(result, expected)