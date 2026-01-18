from itertools import product
import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
def assert_check_nselect_boundary(vals, dtype, method):
    ser = Series(vals, dtype=dtype)
    result = getattr(ser, method)(3)
    expected_idxr = [0, 1, 2] if method == 'nsmallest' else [3, 2, 1]
    expected = ser.loc[expected_idxr]
    tm.assert_series_equal(result, expected)