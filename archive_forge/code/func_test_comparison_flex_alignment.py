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
@pytest.mark.parametrize('values, op', [([False, False, True, False], 'eq'), ([True, True, False, True], 'ne'), ([False, False, True, False], 'le'), ([False, False, False, False], 'lt'), ([False, True, True, False], 'ge'), ([False, True, False, False], 'gt')])
def test_comparison_flex_alignment(self, values, op):
    left = Series([1, 3, 2], index=list('abc'))
    right = Series([2, 2, 2], index=list('bcd'))
    result = getattr(left, op)(right)
    expected = Series(values, index=list('abcd'))
    tm.assert_series_equal(result, expected)