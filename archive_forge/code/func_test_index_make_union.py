import operator
import numpy as np
import pytest
import pandas._libs.sparse as splib
import pandas.util._test_decorators as td
from pandas import Series
import pandas._testing as tm
from pandas.core.arrays.sparse import (
@pytest.mark.parametrize('xloc, xlen, yloc, ylen, eloc, elen', [[[0], [5], [5], [4], [0], [9]], [[0, 10], [5, 5], [2, 17], [5, 2], [0, 10, 17], [7, 5, 2]], [[1], [5], [3], [5], [1], [7]], [[2, 10], [4, 4], [4], [8], [2], [12]], [[0, 5], [3, 5], [0], [7], [0], [10]], [[2, 10], [4, 4], [4, 13], [8, 4], [2], [15]], [[2], [15], [4, 9, 14], [3, 2, 2], [2], [15]], [[0, 10], [3, 3], [5, 15], [2, 2], [0, 5, 10, 15], [3, 2, 3, 2]]])
def test_index_make_union(self, xloc, xlen, yloc, ylen, eloc, elen, test_length):
    xindex = BlockIndex(test_length, xloc, xlen)
    yindex = BlockIndex(test_length, yloc, ylen)
    bresult = xindex.make_union(yindex)
    assert isinstance(bresult, BlockIndex)
    tm.assert_numpy_array_equal(bresult.blocs, np.array(eloc, dtype=np.int32))
    tm.assert_numpy_array_equal(bresult.blengths, np.array(elen, dtype=np.int32))
    ixindex = xindex.to_int_index()
    iyindex = yindex.to_int_index()
    iresult = ixindex.make_union(iyindex)
    assert isinstance(iresult, IntIndex)
    tm.assert_numpy_array_equal(iresult.indices, bresult.to_int_index().indices)