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
@pytest.mark.parametrize('arg, exp', [[[3, 1, 2, 0, 4], [0, 1, 2, 3, 4]], [np.array(list('baaacb'), dtype=object), np.array(list('aaabbc'), dtype=object)], [[], []]])
def test_basic_sort(self, arg, exp):
    result = safe_sort(np.array(arg))
    expected = np.array(exp)
    tm.assert_numpy_array_equal(result, expected)