import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_join_multi_return_indexers():
    midx1 = MultiIndex.from_product([[1, 2], [3, 4], [5, 6]], names=['a', 'b', 'c'])
    midx2 = MultiIndex.from_product([[1, 2], [3, 4]], names=['a', 'b'])
    result = midx1.join(midx2, return_indexers=False)
    tm.assert_index_equal(result, midx1)