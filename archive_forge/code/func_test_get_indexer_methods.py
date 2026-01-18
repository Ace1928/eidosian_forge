from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_indexer_methods(self):
    mult_idx_1 = MultiIndex.from_product([[-1, 0, 1], [0, 2, 3, 4]])
    mult_idx_2 = MultiIndex.from_product([[0], [1, 3, 4]])
    indexer = mult_idx_1.get_indexer(mult_idx_2)
    expected = np.array([-1, 6, 7], dtype=indexer.dtype)
    tm.assert_almost_equal(expected, indexer)
    backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method='backfill')
    expected = np.array([5, 6, 7], dtype=backfill_indexer.dtype)
    tm.assert_almost_equal(expected, backfill_indexer)
    backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method='bfill')
    expected = np.array([5, 6, 7], dtype=backfill_indexer.dtype)
    tm.assert_almost_equal(expected, backfill_indexer)
    pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method='pad')
    expected = np.array([4, 6, 7], dtype=pad_indexer.dtype)
    tm.assert_almost_equal(expected, pad_indexer)
    pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method='ffill')
    expected = np.array([4, 6, 7], dtype=pad_indexer.dtype)
    tm.assert_almost_equal(expected, pad_indexer)