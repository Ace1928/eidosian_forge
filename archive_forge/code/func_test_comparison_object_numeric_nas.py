import datetime
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_comparison_object_numeric_nas(self, comparison_op):
    ser = Series(np.random.default_rng(2).standard_normal(10), dtype=object)
    shifted = ser.shift(2)
    func = comparison_op
    result = func(ser, shifted)
    expected = func(ser.astype(float), shifted.astype(float))
    tm.assert_series_equal(result, expected)