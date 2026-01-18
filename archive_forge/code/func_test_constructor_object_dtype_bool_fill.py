import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_constructor_object_dtype_bool_fill(self):
    data = [False, 0, 100.0, 0.0]
    arr = SparseArray(data, dtype=object, fill_value=False)
    assert arr.dtype == SparseDtype(object, False)
    assert arr.fill_value is False
    arr_expected = np.array(data, dtype=object)
    it = (type(x) == type(y) and x == y for x, y in zip(arr, arr_expected))
    assert np.fromiter(it, dtype=np.bool_).all()