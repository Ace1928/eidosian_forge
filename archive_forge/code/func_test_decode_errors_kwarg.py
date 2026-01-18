from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
def test_decode_errors_kwarg():
    ser = Series([b'a', b'b', b'a\x9d'])
    msg = "'charmap' codec can't decode byte 0x9d in position 1: character maps to <undefined>"
    with pytest.raises(UnicodeDecodeError, match=msg):
        ser.str.decode('cp1252')
    result = ser.str.decode('cp1252', 'ignore')
    expected = ser.map(lambda x: x.decode('cp1252', 'ignore')).astype(object)
    tm.assert_series_equal(result, expected)