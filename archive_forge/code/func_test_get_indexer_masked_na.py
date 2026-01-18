import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('val', [4, 2])
def test_get_indexer_masked_na(self, any_numeric_ea_and_arrow_dtype, val):
    idx = Index([1, 2, NA, 3, val], dtype=any_numeric_ea_and_arrow_dtype)
    result = idx.get_indexer_for([1, NA, 5])
    expected = np.array([0, 2, -1])
    tm.assert_numpy_array_equal(result, expected, check_dtype=False)