import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_constructor_bool_fill_value(self):
    arr = SparseArray([True, False, True], dtype=None)
    assert arr.dtype == SparseDtype(np.bool_)
    assert not arr.fill_value
    arr = SparseArray([True, False, True], dtype=np.bool_)
    assert arr.dtype == SparseDtype(np.bool_)
    assert not arr.fill_value
    arr = SparseArray([True, False, True], dtype=np.bool_, fill_value=True)
    assert arr.dtype == SparseDtype(np.bool_, True)
    assert arr.fill_value