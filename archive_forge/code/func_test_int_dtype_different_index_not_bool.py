import operator
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_int_dtype_different_index_not_bool(self):
    df1 = DataFrame([1, 2, 3], index=[10, 11, 23], columns=['a'])
    df2 = DataFrame([10, 20, 30], index=[11, 10, 23], columns=['a'])
    result = np.bitwise_xor(df1, df2)
    expected = DataFrame([21, 8, 29], index=[10, 11, 23], columns=['a'])
    tm.assert_frame_equal(result, expected)
    result = df1 ^ df2
    tm.assert_frame_equal(result, expected)