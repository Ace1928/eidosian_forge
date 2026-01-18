import numpy as np
import pytest
from pandas.core.dtypes import dtypes
from pandas.core.dtypes.common import is_extension_array_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
def test_astype_no_copy():
    arr = DummyArray(np.array([1, 2, 3], dtype=np.int64))
    result = arr.astype(arr.dtype, copy=False)
    assert arr is result
    result = arr.astype(arr.dtype)
    assert arr is not result