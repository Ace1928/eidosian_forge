from collections import defaultdict
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
import pandas.core.common as com
from pandas.core.sorting import (
@pytest.mark.parametrize('codes_list, shape', [[[np.tile([0, 1, 2, 3, 0, 1, 2, 3], 100).astype(np.int64), np.tile([0, 2, 4, 3, 0, 1, 2, 3], 100).astype(np.int64), np.tile([5, 1, 0, 2, 3, 0, 5, 4], 100).astype(np.int64)], (4, 5, 6)], [[np.tile(np.arange(10000, dtype=np.int64), 5), np.tile(np.arange(10000, dtype=np.int64), 5)], (10000, 10000)]])
def test_decons(codes_list, shape):
    group_index = get_group_index(codes_list, shape, sort=True, xnull=True)
    codes_list2 = _decons_group_index(group_index, shape)
    for a, b in zip(codes_list, codes_list2):
        tm.assert_numpy_array_equal(a, b)