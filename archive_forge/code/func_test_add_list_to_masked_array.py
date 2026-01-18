from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
@pytest.mark.parametrize('val, dtype', [(3, 'Int64'), (3.5, 'Float64')])
def test_add_list_to_masked_array(self, val, dtype):
    ser = Series([1, None, 3], dtype='Int64')
    result = ser + [1, None, val]
    expected = Series([2, None, 3 + val], dtype=dtype)
    tm.assert_series_equal(result, expected)
    result = [1, None, val] + ser
    tm.assert_series_equal(result, expected)