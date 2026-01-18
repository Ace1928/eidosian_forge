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
@pytest.mark.parametrize('verify', [True, False])
@pytest.mark.parametrize('codes, exp_codes', [[[0, 1, 1, 2, 3, 0, -1, 4], [3, 1, 1, 2, 0, 3, -1, 4]], [[], []]])
def test_codes(self, verify, codes, exp_codes):
    values = np.array([3, 1, 2, 0, 4])
    expected = np.array([0, 1, 2, 3, 4])
    result, result_codes = safe_sort(values, codes, use_na_sentinel=True, verify=verify)
    expected_codes = np.array(exp_codes, dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)
    tm.assert_numpy_array_equal(result_codes, expected_codes)