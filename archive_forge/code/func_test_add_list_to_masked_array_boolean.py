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
def test_add_list_to_masked_array_boolean(self, request):
    warning = UserWarning if request.node.callspec.id == 'numexpr' and NUMEXPR_INSTALLED else None
    ser = Series([True, None, False], dtype='boolean')
    with tm.assert_produces_warning(warning):
        result = ser + [True, None, True]
    expected = Series([True, None, True], dtype='boolean')
    tm.assert_series_equal(result, expected)
    with tm.assert_produces_warning(warning):
        result = [True, None, True] + ser
    tm.assert_series_equal(result, expected)