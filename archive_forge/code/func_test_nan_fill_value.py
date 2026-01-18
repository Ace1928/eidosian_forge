import numpy as np
import pytest
from pandas import (
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('raw_data,max_expected,min_expected', [(np.arange(5.0), [4], [0]), (-np.arange(5.0), [0], [-4]), (np.array([0, 1, 2, np.nan, 4]), [4], [0]), (np.array([np.nan] * 5), [np.nan], [np.nan]), (np.array([]), [np.nan], [np.nan])])
def test_nan_fill_value(self, raw_data, max_expected, min_expected):
    arr = SparseArray(raw_data)
    max_result = arr.max()
    min_result = arr.min()
    assert max_result in max_expected
    assert min_result in min_expected
    max_result = arr.max(skipna=False)
    min_result = arr.min(skipna=False)
    if np.isnan(raw_data).any():
        assert np.isnan(max_result)
        assert np.isnan(min_result)
    else:
        assert max_result in max_expected
        assert min_result in min_expected