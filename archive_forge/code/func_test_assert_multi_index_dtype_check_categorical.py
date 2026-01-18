import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('check_categorical', [True, False])
def test_assert_multi_index_dtype_check_categorical(check_categorical):
    idx1 = MultiIndex.from_arrays([Categorical(np.array([1, 2], dtype=np.uint64))])
    idx2 = MultiIndex.from_arrays([Categorical(np.array([1, 2], dtype=np.int64))])
    if check_categorical:
        with pytest.raises(AssertionError, match='^MultiIndex level \\[0\\] are different'):
            tm.assert_index_equal(idx1, idx2, check_categorical=check_categorical)
    else:
        tm.assert_index_equal(idx1, idx2, check_categorical=check_categorical)