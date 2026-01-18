import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_float_same_index_with_nans(self, kind, mix, all_arithmetic_functions, request):
    op = all_arithmetic_functions
    values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
    rvalues = np.array([np.nan, 2, 3, 4, np.nan, 0, 1, 3, 2, np.nan])
    a = SparseArray(values, kind=kind)
    b = SparseArray(rvalues, kind=kind)
    self._check_numeric_ops(a, b, values, rvalues, mix, op)