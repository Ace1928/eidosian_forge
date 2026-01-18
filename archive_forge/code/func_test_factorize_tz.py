import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_factorize_tz(self, tz_naive_fixture, index_or_series):
    tz = tz_naive_fixture
    base = date_range('2016-11-05', freq='h', periods=100, tz=tz)
    idx = base.repeat(5)
    exp_arr = np.arange(100, dtype=np.intp).repeat(5)
    obj = index_or_series(idx)
    arr, res = obj.factorize()
    tm.assert_numpy_array_equal(arr, exp_arr)
    expected = base._with_freq(None)
    tm.assert_index_equal(res, expected)
    assert res.freq == expected.freq