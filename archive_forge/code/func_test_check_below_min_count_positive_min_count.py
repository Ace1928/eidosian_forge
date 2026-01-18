from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@pytest.mark.parametrize('mask', [None, np.array([False, False, True]), np.array([True] + 9 * [False])])
@pytest.mark.parametrize('min_count, expected_result', [(1, False), (101, True)])
def test_check_below_min_count_positive_min_count(mask, min_count, expected_result):
    shape = (10, 10)
    result = nanops.check_below_min_count(shape, mask, min_count)
    assert result == expected_result