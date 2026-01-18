from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@pytest.mark.parametrize('min_count', [-1, 0])
def test_check_below_min_count_negative_or_zero_min_count(min_count):
    result = nanops.check_below_min_count((21, 37), None, min_count)
    expected_result = False
    assert result == expected_result