from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_indexer_crossing_levels(self):
    mult_idx_1 = MultiIndex.from_product([[1, 2]] * 4)
    mult_idx_2 = MultiIndex.from_tuples([(1, 3, 2, 2), (2, 3, 2, 2)])
    assert mult_idx_1[7] < mult_idx_2[0] < mult_idx_1[8]
    assert mult_idx_1[-1] < mult_idx_2[1]
    indexer = mult_idx_1.get_indexer(mult_idx_2)
    expected = np.array([-1, -1], dtype=indexer.dtype)
    tm.assert_almost_equal(expected, indexer)
    backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method='bfill')
    expected = np.array([8, -1], dtype=backfill_indexer.dtype)
    tm.assert_almost_equal(expected, backfill_indexer)
    pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method='ffill')
    expected = np.array([7, 15], dtype=pad_indexer.dtype)
    tm.assert_almost_equal(expected, pad_indexer)