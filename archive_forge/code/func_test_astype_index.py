import numpy as np
import pytest
from pandas.core.dtypes.generic import ABCIndex
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import (
@pytest.mark.parametrize('dropna', [True, False])
def test_astype_index(all_data, dropna):
    all_data = all_data[:10]
    if dropna:
        other = all_data[~all_data.isna()]
    else:
        other = all_data
    dtype = all_data.dtype
    idx = pd.Index(np.array(other))
    assert isinstance(idx, ABCIndex)
    result = idx.astype(dtype)
    expected = idx.astype(object).astype(dtype)
    tm.assert_index_equal(result, expected)