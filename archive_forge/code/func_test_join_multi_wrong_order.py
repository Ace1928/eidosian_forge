import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_join_multi_wrong_order():
    midx1 = MultiIndex.from_product([[1, 2], [3, 4]], names=['a', 'b'])
    midx2 = MultiIndex.from_product([[1, 2], [3, 4]], names=['b', 'a'])
    join_idx, lidx, ridx = midx1.join(midx2, return_indexers=True)
    exp_ridx = np.array([-1, -1, -1, -1], dtype=np.intp)
    tm.assert_index_equal(midx1, join_idx)
    assert lidx is None
    tm.assert_numpy_array_equal(ridx, exp_ridx)