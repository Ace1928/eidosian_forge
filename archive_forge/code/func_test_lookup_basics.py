import operator
import numpy as np
import pytest
import pandas._libs.sparse as splib
import pandas.util._test_decorators as td
from pandas import Series
import pandas._testing as tm
from pandas.core.arrays.sparse import (
@pytest.mark.parametrize('idx, expected', [[0, -1], [5, 0], [7, 2], [8, -1], [9, -1], [10, -1], [11, -1], [12, 3], [17, 8], [18, -1]])
def test_lookup_basics(self, idx, expected):
    bindex = BlockIndex(20, [5, 12], [3, 6])
    assert bindex.lookup(idx) == expected
    iindex = bindex.to_int_index()
    assert iindex.lookup(idx) == expected