from itertools import product
import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['datetime64[ns]', 'timedelta64[ns]'])
def test_nlargest_boundary_datetimelike(self, nselect_method, dtype):
    dtype_info = np.iinfo('int64')
    min_val, max_val = (dtype_info.min, dtype_info.max)
    vals = [min_val + 1, min_val + 2, max_val - 1, max_val, min_val]
    assert_check_nselect_boundary(vals, dtype, nselect_method)