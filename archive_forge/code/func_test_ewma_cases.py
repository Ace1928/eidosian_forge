import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ewma_cases(adjust, ignore_na):
    s = Series([1.0, 2.0, 4.0, 8.0])
    if adjust:
        expected = Series([1.0, 1.6, 2.736842, 4.923077])
    else:
        expected = Series([1.0, 1.333333, 2.222222, 4.148148])
    result = s.ewm(com=2.0, adjust=adjust, ignore_na=ignore_na).mean()
    tm.assert_series_equal(result, expected)