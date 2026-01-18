import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_get_loc_masked_na(self, any_numeric_ea_and_arrow_dtype):
    idx = Index([1, 2, NA], dtype=any_numeric_ea_and_arrow_dtype)
    result = idx.get_loc(NA)
    assert result == 2
    idx = Index([1, 2, NA, NA], dtype=any_numeric_ea_and_arrow_dtype)
    result = idx.get_loc(NA)
    tm.assert_numpy_array_equal(result, np.array([False, False, True, True]))
    idx = Index([1, 2, 3], dtype=any_numeric_ea_and_arrow_dtype)
    with pytest.raises(KeyError, match='NA'):
        idx.get_loc(NA)