import numpy as np
from pandas import (
import pandas._testing as tm
def test_arithmetic_zero_freq(self):
    tdi = timedelta_range(0, periods=100, freq='ns')
    result = tdi / 2
    assert result.freq is None
    expected = tdi[:50].repeat(2)
    tm.assert_index_equal(result, expected)
    result2 = tdi // 2
    assert result2.freq is None
    expected2 = expected
    tm.assert_index_equal(result2, expected2)
    result3 = tdi * 0
    assert result3.freq is None
    expected3 = tdi[:1].repeat(100)
    tm.assert_index_equal(result3, expected3)