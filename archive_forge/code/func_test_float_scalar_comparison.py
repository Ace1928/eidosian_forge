import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_float_scalar_comparison(self, kind):
    values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
    a = SparseArray(values, kind=kind)
    self._check_comparison_ops(a, 1, values, 1)
    self._check_comparison_ops(a, 0, values, 0)
    self._check_comparison_ops(a, 3, values, 3)
    a = SparseArray(values, kind=kind, fill_value=0)
    self._check_comparison_ops(a, 1, values, 1)
    self._check_comparison_ops(a, 0, values, 0)
    self._check_comparison_ops(a, 3, values, 3)
    a = SparseArray(values, kind=kind, fill_value=2)
    self._check_comparison_ops(a, 1, values, 1)
    self._check_comparison_ops(a, 0, values, 0)
    self._check_comparison_ops(a, 3, values, 3)