import operator
import numpy as np
import pytest
import pandas._libs.sparse as splib
import pandas.util._test_decorators as td
from pandas import Series
import pandas._testing as tm
from pandas.core.arrays.sparse import (
@pytest.mark.parametrize('i', [5, 10, 100, 101])
def test_make_block_boundary(self, i):
    idx = make_sparse_index(i, np.arange(0, i, 2, dtype=np.int32), kind='block')
    exp = np.arange(0, i, 2, dtype=np.int32)
    tm.assert_numpy_array_equal(idx.blocs, exp)
    tm.assert_numpy_array_equal(idx.blengths, np.ones(len(exp), dtype=np.int32))