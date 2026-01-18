import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_int_array(self, kind, mix, all_arithmetic_functions):
    op = all_arithmetic_functions
    dtype = np.int64
    values = np.array([0, 1, 2, 0, 0, 0, 1, 2, 1, 0], dtype=dtype)
    rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=dtype)
    a = SparseArray(values, dtype=dtype, kind=kind)
    assert a.dtype == SparseDtype(dtype)
    b = SparseArray(rvalues, dtype=dtype, kind=kind)
    assert b.dtype == SparseDtype(dtype)
    self._check_numeric_ops(a, b, values, rvalues, mix, op)
    self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)
    a = SparseArray(values, fill_value=0, dtype=dtype, kind=kind)
    assert a.dtype == SparseDtype(dtype)
    b = SparseArray(rvalues, dtype=dtype, kind=kind)
    assert b.dtype == SparseDtype(dtype)
    self._check_numeric_ops(a, b, values, rvalues, mix, op)
    a = SparseArray(values, fill_value=0, dtype=dtype, kind=kind)
    assert a.dtype == SparseDtype(dtype)
    b = SparseArray(rvalues, fill_value=0, dtype=dtype, kind=kind)
    assert b.dtype == SparseDtype(dtype)
    self._check_numeric_ops(a, b, values, rvalues, mix, op)
    a = SparseArray(values, fill_value=1, dtype=dtype, kind=kind)
    assert a.dtype == SparseDtype(dtype, fill_value=1)
    b = SparseArray(rvalues, fill_value=2, dtype=dtype, kind=kind)
    assert b.dtype == SparseDtype(dtype, fill_value=2)
    self._check_numeric_ops(a, b, values, rvalues, mix, op)