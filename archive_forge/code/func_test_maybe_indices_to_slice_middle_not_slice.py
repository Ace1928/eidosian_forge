import numpy as np
import pytest
from pandas._libs import (
from pandas.compat import IS64
from pandas import Index
import pandas._testing as tm
@pytest.mark.parametrize('case', [[14, 12, 10, 12], [12, 12, 11, 10], [10, 11, 12, 11]])
def test_maybe_indices_to_slice_middle_not_slice(self, case):
    target = np.arange(100)
    indices = np.array(case, dtype=np.intp)
    maybe_slice = lib.maybe_indices_to_slice(indices, len(target))
    assert not isinstance(maybe_slice, slice)
    tm.assert_numpy_array_equal(maybe_slice, indices)
    tm.assert_numpy_array_equal(target[indices], target[maybe_slice])