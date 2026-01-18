import numpy as np
from pandas import (
import pandas._testing as tm
def test_infer_objects_bytes(self):
    ser = Series([b'a'], dtype='bytes')
    expected = ser.copy()
    result = ser.infer_objects()
    tm.assert_series_equal(result, expected)