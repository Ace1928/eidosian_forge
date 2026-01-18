from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_indexer_three_or_more_levels(self):
    mult_idx_1 = MultiIndex.from_product([[1, 3], [2, 4, 6], [5, 7]])
    mult_idx_2 = MultiIndex.from_tuples([(1, 1, 8), (1, 5, 9), (1, 6, 7), (2, 1, 6), (2, 7, 7), (2, 7, 8), (3, 6, 8)])
    assert mult_idx_1.is_monotonic_increasing
    assert mult_idx_1.is_unique
    assert mult_idx_2.is_monotonic_increasing
    assert mult_idx_2.is_unique
    assert mult_idx_2[0] < mult_idx_1[0]
    assert mult_idx_1[3] < mult_idx_2[1] < mult_idx_1[4]
    assert mult_idx_1[5] == mult_idx_2[2]
    assert mult_idx_1[5] < mult_idx_2[3] < mult_idx_1[6]
    assert mult_idx_1[5] < mult_idx_2[4] < mult_idx_1[6]
    assert mult_idx_1[5] < mult_idx_2[5] < mult_idx_1[6]
    assert mult_idx_1[-1] < mult_idx_2[6]
    indexer_no_fill = mult_idx_1.get_indexer(mult_idx_2)
    expected = np.array([-1, -1, 5, -1, -1, -1, -1], dtype=indexer_no_fill.dtype)
    tm.assert_almost_equal(expected, indexer_no_fill)
    indexer_backfilled = mult_idx_1.get_indexer(mult_idx_2, method='backfill')
    expected = np.array([0, 4, 5, 6, 6, 6, -1], dtype=indexer_backfilled.dtype)
    tm.assert_almost_equal(expected, indexer_backfilled)
    indexer_padded = mult_idx_1.get_indexer(mult_idx_2, method='pad')
    expected = np.array([-1, 3, 5, 5, 5, 5, 11], dtype=indexer_padded.dtype)
    tm.assert_almost_equal(expected, indexer_padded)
    assert mult_idx_2[0] < mult_idx_1[0] < mult_idx_2[1]
    assert mult_idx_2[0] < mult_idx_1[1] < mult_idx_2[1]
    assert mult_idx_2[0] < mult_idx_1[2] < mult_idx_2[1]
    assert mult_idx_2[0] < mult_idx_1[3] < mult_idx_2[1]
    assert mult_idx_2[1] < mult_idx_1[4] < mult_idx_2[2]
    assert mult_idx_2[2] == mult_idx_1[5]
    assert mult_idx_2[5] < mult_idx_1[6] < mult_idx_2[6]
    assert mult_idx_2[5] < mult_idx_1[7] < mult_idx_2[6]
    assert mult_idx_2[5] < mult_idx_1[8] < mult_idx_2[6]
    assert mult_idx_2[5] < mult_idx_1[9] < mult_idx_2[6]
    assert mult_idx_2[5] < mult_idx_1[10] < mult_idx_2[6]
    assert mult_idx_2[5] < mult_idx_1[11] < mult_idx_2[6]
    indexer = mult_idx_2.get_indexer(mult_idx_1)
    expected = np.array([-1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1], dtype=indexer.dtype)
    tm.assert_almost_equal(expected, indexer)
    backfill_indexer = mult_idx_2.get_indexer(mult_idx_1, method='bfill')
    expected = np.array([1, 1, 1, 1, 2, 2, 6, 6, 6, 6, 6, 6], dtype=backfill_indexer.dtype)
    tm.assert_almost_equal(expected, backfill_indexer)
    pad_indexer = mult_idx_2.get_indexer(mult_idx_1, method='pad')
    expected = np.array([0, 0, 0, 0, 1, 2, 5, 5, 5, 5, 5, 5], dtype=pad_indexer.dtype)
    tm.assert_almost_equal(expected, pad_indexer)