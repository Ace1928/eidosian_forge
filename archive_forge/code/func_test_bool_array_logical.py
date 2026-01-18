import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('fill_value', [True, False, np.nan])
def test_bool_array_logical(self, kind, fill_value):
    values = np.array([True, False, True, False, True, True], dtype=np.bool_)
    rvalues = np.array([True, False, False, True, False, True], dtype=np.bool_)
    a = SparseArray(values, kind=kind, dtype=np.bool_, fill_value=fill_value)
    b = SparseArray(rvalues, kind=kind, dtype=np.bool_, fill_value=fill_value)
    self._check_logical_ops(a, b, values, rvalues)