import numpy as np
import pytest
from pandas.core.dtypes.generic import ABCIndex
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import (
@pytest.mark.parametrize('dropna', [True, False])
def test_construct_index(all_data, dropna):
    all_data = all_data[:10]
    if dropna:
        other = np.array(all_data[~all_data.isna()])
    else:
        other = all_data
    result = pd.Index(pd.array(other, dtype=all_data.dtype))
    expected = pd.Index(other, dtype=all_data.dtype)
    assert all_data.dtype == expected.dtype
    tm.assert_index_equal(result, expected)