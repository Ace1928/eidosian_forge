import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
def test_get_loc_masked_duplicated_na(self):
    idx = Index([1, 2, NA, NA], dtype='Int64')
    result = idx.get_loc(NA)
    expected = np.array([False, False, True, True])
    tm.assert_numpy_array_equal(result, expected)