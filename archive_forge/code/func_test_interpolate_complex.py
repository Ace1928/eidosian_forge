import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_interpolate_complex(self):
    ser = Series([complex('1+1j'), float('nan'), complex('2+2j')])
    assert ser.dtype.kind == 'c'
    res = ser.interpolate()
    expected = Series([ser[0], ser[0] * 1.5, ser[2]])
    tm.assert_series_equal(res, expected)
    df = ser.to_frame()
    res = df.interpolate()
    expected = expected.to_frame()
    tm.assert_frame_equal(res, expected)