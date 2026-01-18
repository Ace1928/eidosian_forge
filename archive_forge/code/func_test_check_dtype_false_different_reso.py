import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['datetime64', 'timedelta64'])
def test_check_dtype_false_different_reso(dtype):
    ser_s = Series([1000213, 2131232, 21312331]).astype(f'{dtype}[s]')
    ser_ms = ser_s.astype(f'{dtype}[ms]')
    with pytest.raises(AssertionError, match='Attributes of Series are different'):
        tm.assert_series_equal(ser_s, ser_ms)
    tm.assert_series_equal(ser_ms, ser_s, check_dtype=False)
    ser_ms -= Series([1, 1, 1]).astype(f'{dtype}[ms]')
    with pytest.raises(AssertionError, match='Series are different'):
        tm.assert_series_equal(ser_s, ser_ms)
    with pytest.raises(AssertionError, match='Series are different'):
        tm.assert_series_equal(ser_s, ser_ms, check_dtype=False)