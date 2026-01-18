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
def test_add_with_duplicate_index(self):
    s1 = Series([1, 2], index=[1, 1])
    s2 = Series([10, 10], index=[1, 2])
    result = s1 + s2
    expected = Series([11, 12, np.nan], index=[1, 1, 2])
    tm.assert_series_equal(result, expected)