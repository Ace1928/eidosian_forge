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
@pytest.mark.parametrize('values, op, fill_value', [([False, False, True, True], 'eq', 2), ([True, True, False, False], 'ne', 2), ([False, False, True, True], 'le', 0), ([False, False, False, True], 'lt', 0), ([True, True, True, False], 'ge', 0), ([True, True, False, False], 'gt', 0)])
def test_comparison_flex_alignment_fill(self, values, op, fill_value):
    left = Series([1, 3, 2], index=list('abc'))
    right = Series([2, 2, 2], index=list('bcd'))
    result = getattr(left, op)(right, fill_value=fill_value)
    expected = Series(values, index=list('abcd'))
    tm.assert_series_equal(result, expected)