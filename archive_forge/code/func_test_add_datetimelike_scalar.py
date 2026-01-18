from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import Timedelta
import pandas._testing as tm
from pandas.core.arrays import (
def test_add_datetimelike_scalar(self, tda, tz_naive_fixture):
    ts = pd.Timestamp('2016-01-01', tz=tz_naive_fixture).as_unit('ns')
    expected = tda.as_unit('ns') + ts
    res = tda + ts
    tm.assert_extension_array_equal(res, expected)
    res = ts + tda
    tm.assert_extension_array_equal(res, expected)
    ts += Timedelta(1)
    exp_values = tda._ndarray + ts.asm8
    expected = DatetimeArray._simple_new(exp_values, dtype=exp_values.dtype).tz_localize('UTC').tz_convert(ts.tz)
    result = tda + ts
    tm.assert_extension_array_equal(result, expected)
    result = ts + tda
    tm.assert_extension_array_equal(result, expected)