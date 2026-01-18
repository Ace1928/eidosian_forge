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
@pytest.mark.slow
def test_int64_overflow_check_sum_col(self, left_right):
    left, right = left_right
    out = merge(left, right, how='outer')
    assert len(out) == len(left)
    tm.assert_series_equal(out['left'], -out['right'], check_names=False)
    result = out.iloc[:, :-2].sum(axis=1)
    tm.assert_series_equal(out['left'], result, check_names=False)
    assert result.name is None