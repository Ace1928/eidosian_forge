import numpy as np
from pandas import (
from pandas.core.arrays import (
def test_empty_pandas_array(self):
    arr = NumpyExtensionArray(np.array([1, 2]))
    dtype = arr.dtype
    shape = (3, 9)
    result = NumpyExtensionArray._empty(shape, dtype=dtype)
    assert isinstance(result, NumpyExtensionArray)
    assert result.dtype == dtype
    assert result.shape == shape