import numpy as np
import pytest
from pandas._libs import index as libindex
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('idx_values,idx_non_unique', [([np.nan, 100, 200, 100], [np.nan, 100]), ([np.nan, 100.0, 200.0, 100.0], [np.nan, 100.0])])
def test_get_indexer_non_unique_int_index(self, idx_values, idx_non_unique):
    indexes, missing = Index(idx_values).get_indexer_non_unique(Index([np.nan]))
    tm.assert_numpy_array_equal(np.array([0], dtype=np.intp), indexes)
    tm.assert_numpy_array_equal(np.array([], dtype=np.intp), missing)
    indexes, missing = Index(idx_values).get_indexer_non_unique(Index(idx_non_unique))
    tm.assert_numpy_array_equal(np.array([0, 1, 3], dtype=np.intp), indexes)
    tm.assert_numpy_array_equal(np.array([], dtype=np.intp), missing)