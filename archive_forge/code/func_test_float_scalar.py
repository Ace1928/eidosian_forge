import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('scalar', [0, 1, 3])
@pytest.mark.parametrize('fill_value', [None, 0, 2])
def test_float_scalar(self, kind, mix, all_arithmetic_functions, fill_value, scalar, request):
    op = all_arithmetic_functions
    values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
    a = SparseArray(values, kind=kind, fill_value=fill_value)
    self._check_numeric_ops(a, scalar, values, scalar, mix, op)