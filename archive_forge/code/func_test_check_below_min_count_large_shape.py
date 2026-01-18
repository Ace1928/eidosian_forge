from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@td.skip_if_windows
@td.skip_if_32bit
@pytest.mark.parametrize('min_count, expected_result', [(1, False), (2812191852, True)])
def test_check_below_min_count_large_shape(min_count, expected_result):
    shape = (2244367, 1253)
    result = nanops.check_below_min_count(shape, mask=None, min_count=min_count)
    assert result == expected_result