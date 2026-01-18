from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_nan_int_timedelta_sum(self):
    df = DataFrame({'A': Series([1, 2, NaT], dtype='timedelta64[ns]'), 'B': Series([1, 2, np.nan], dtype='Int64')})
    expected = Series({'A': Timedelta(3), 'B': 3})
    result = df.sum()
    tm.assert_series_equal(result, expected)