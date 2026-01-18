import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data1,data2', [(0.12345, 0.12346), (0.1235, 0.1236)])
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'Float32'])
@pytest.mark.parametrize('decimals', [0, 1, 2, 3, 5, 10])
def test_less_precise(data1, data2, dtype, decimals):
    rtol = 10 ** (-decimals)
    s1 = Series([data1], dtype=dtype)
    s2 = Series([data2], dtype=dtype)
    if decimals in (5, 10) or (decimals >= 3 and abs(data1 - data2) >= 0.0005):
        msg = 'Series values are different'
        with pytest.raises(AssertionError, match=msg):
            tm.assert_series_equal(s1, s2, rtol=rtol)
    else:
        _assert_series_equal_both(s1, s2, rtol=rtol)