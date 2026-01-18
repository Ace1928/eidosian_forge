import numpy as np
import pytest
from pandas._libs import (
from pandas.compat import IS64
from pandas import Index
import pandas._testing as tm
def test_maybe_indices_to_slice_left_edge(self):
    target = np.arange(100)
    indices = np.array([], dtype=np.intp)
    maybe_slice = lib.maybe_indices_to_slice(indices, len(target))
    assert isinstance(maybe_slice, slice)
    tm.assert_numpy_array_equal(target[indices], target[maybe_slice])