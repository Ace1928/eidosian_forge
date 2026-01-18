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
def test_mask_div_propagate_na_for_non_na_dtype(self):
    ser1 = Series([15, pd.NA, 5, 4], dtype='Int64')
    ser2 = Series([15, 5, np.nan, 4])
    result = ser1 / ser2
    expected = Series([1.0, pd.NA, pd.NA, 1.0], dtype='Float64')
    tm.assert_series_equal(result, expected)
    result = ser2 / ser1
    tm.assert_series_equal(result, expected)