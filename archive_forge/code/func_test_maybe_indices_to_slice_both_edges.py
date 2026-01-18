import numpy as np
import pytest
from pandas._libs import (
from pandas.compat import IS64
from pandas import Index
import pandas._testing as tm
@pytest.mark.parametrize('step', [1, 2, 4, 5, 8, 9])
def test_maybe_indices_to_slice_both_edges(self, step):
    target = np.arange(10)
    indices = np.arange(0, 9, step, dtype=np.intp)
    maybe_slice = lib.maybe_indices_to_slice(indices, len(target))
    assert isinstance(maybe_slice, slice)
    tm.assert_numpy_array_equal(target[indices], target[maybe_slice])
    indices = indices[::-1]
    maybe_slice = lib.maybe_indices_to_slice(indices, len(target))
    assert isinstance(maybe_slice, slice)
    tm.assert_numpy_array_equal(target[indices], target[maybe_slice])