import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('listtype', [list, tuple, Series, np.array])
@pytest.mark.parametrize('tolerance, expected', list(zip([[0.3, 0.3, 0.1], [0.2, 0.1, 0.1], [0.1, 0.5, 0.5]], [[0, 2, -1], [0, -1, -1], [-1, 2, 9]])))
def test_get_indexer_nearest_listlike_tolerance(self, tolerance, expected, listtype):
    index = Index(np.arange(10))
    actual = index.get_indexer([0.2, 1.8, 8.5], method='nearest', tolerance=listtype(tolerance))
    tm.assert_numpy_array_equal(actual, np.array(expected, dtype=np.intp))