from itertools import product
import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
def test_nlargest_boundary_integer(self, nselect_method, any_int_numpy_dtype):
    dtype_info = np.iinfo(any_int_numpy_dtype)
    min_val, max_val = (dtype_info.min, dtype_info.max)
    vals = [min_val, min_val + 1, max_val - 1, max_val]
    assert_check_nselect_boundary(vals, any_int_numpy_dtype, nselect_method)