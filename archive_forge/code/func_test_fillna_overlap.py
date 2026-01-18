import re
import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_fillna_overlap(self):
    s = SparseArray([1, np.nan, np.nan, 3, np.nan])
    res = s.fillna(3)
    exp = np.array([1, 3, 3, 3, 3], dtype=np.float64)
    tm.assert_numpy_array_equal(res.to_dense(), exp)
    s = SparseArray([1, np.nan, np.nan, 3, np.nan], fill_value=0)
    res = s.fillna(3)
    exp = SparseArray([1, 3, 3, 3, 3], fill_value=0, dtype=np.float64)
    tm.assert_sp_array_equal(res, exp)