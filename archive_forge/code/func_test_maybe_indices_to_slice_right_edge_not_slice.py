import numpy as np
import pytest
from pandas._libs import (
from pandas.compat import IS64
from pandas import Index
import pandas._testing as tm
def test_maybe_indices_to_slice_right_edge_not_slice(self):
    target = np.arange(100)
    indices = np.array([97, 98, 99, 100], dtype=np.intp)
    maybe_slice = lib.maybe_indices_to_slice(indices, len(target))
    assert not isinstance(maybe_slice, slice)
    tm.assert_numpy_array_equal(maybe_slice, indices)
    msg = 'index 100 is out of bounds for axis (0|1) with size 100'
    with pytest.raises(IndexError, match=msg):
        target[indices]
    with pytest.raises(IndexError, match=msg):
        target[maybe_slice]
    indices = np.array([100, 99, 98, 97], dtype=np.intp)
    maybe_slice = lib.maybe_indices_to_slice(indices, len(target))
    assert not isinstance(maybe_slice, slice)
    tm.assert_numpy_array_equal(maybe_slice, indices)
    with pytest.raises(IndexError, match=msg):
        target[indices]
    with pytest.raises(IndexError, match=msg):
        target[maybe_slice]