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
def test_comparison_frozenset(self):
    ser = Series([frozenset([1]), frozenset([1, 2])])
    result = ser == frozenset([1])
    expected = Series([True, False])
    tm.assert_series_equal(result, expected)