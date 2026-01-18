import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
def test_apply_empty_integer_series_with_datetime_index(by_row):
    s = Series([], index=date_range(start='2018-01-01', periods=0), dtype=int)
    result = s.apply(lambda x: x, by_row=by_row)
    tm.assert_series_equal(result, s)