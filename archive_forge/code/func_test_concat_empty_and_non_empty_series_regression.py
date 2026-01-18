import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_concat_empty_and_non_empty_series_regression(self):
    s1 = Series([1])
    s2 = Series([], dtype=object)
    expected = s1
    msg = 'The behavior of array concatenation with empty entries is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = concat([s1, s2])
    tm.assert_series_equal(result, expected)