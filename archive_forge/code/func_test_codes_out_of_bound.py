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
def test_codes_out_of_bound(self):
    values = np.array([3, 1, 2, 0, 4])
    expected = np.array([0, 1, 2, 3, 4])
    codes = [0, 101, 102, 2, 3, 0, 99, 4]
    result, result_codes = safe_sort(values, codes, use_na_sentinel=True)
    expected_codes = np.array([3, -1, -1, 2, 0, 3, -1, 4], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)
    tm.assert_numpy_array_equal(result_codes, expected_codes)