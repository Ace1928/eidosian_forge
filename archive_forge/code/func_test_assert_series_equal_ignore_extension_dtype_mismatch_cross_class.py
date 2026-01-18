import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_assert_series_equal_ignore_extension_dtype_mismatch_cross_class():
    left = Series([1, 2, 3], dtype='Int64')
    right = Series([1, 2, 3], dtype='int64')
    tm.assert_series_equal(left, right, check_dtype=False)