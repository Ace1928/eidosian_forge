import numpy as np
import pytest
from pandas._libs import (
from pandas.compat import IS64
from pandas import Index
import pandas._testing as tm
@pytest.mark.parametrize('start, end', [(2, 10), (5, 25), (65, 97)])
@pytest.mark.parametrize('step', [1, 2, 4, 20])
def test_maybe_indices_to_slice_middle(self, start, end, step):
    target = np.arange(100)
    indices = np.arange(start, end, step, dtype=np.intp)
    maybe_slice = lib.maybe_indices_to_slice(indices, len(target))
    assert isinstance(maybe_slice, slice)
    tm.assert_numpy_array_equal(target[indices], target[maybe_slice])
    indices = indices[::-1]
    maybe_slice = lib.maybe_indices_to_slice(indices, len(target))
    assert isinstance(maybe_slice, slice)
    tm.assert_numpy_array_equal(target[indices], target[maybe_slice])