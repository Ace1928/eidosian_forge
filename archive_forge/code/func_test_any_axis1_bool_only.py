from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_any_axis1_bool_only(self):
    df = DataFrame({'A': [True, False], 'B': [1, 2]})
    result = df.any(axis=1, bool_only=True)
    expected = Series([True, False])
    tm.assert_series_equal(result, expected)