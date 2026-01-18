import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_describe_bool_frame(self):
    df = DataFrame({'bool_data_1': [False, False, True, True], 'bool_data_2': [False, True, True, True]})
    result = df.describe()
    expected = DataFrame({'bool_data_1': [4, 2, False, 2], 'bool_data_2': [4, 2, True, 3]}, index=['count', 'unique', 'top', 'freq'])
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'bool_data': [False, False, True, True, False], 'int_data': [0, 1, 2, 3, 4]})
    result = df.describe()
    expected = DataFrame({'int_data': [5, 2, df.int_data.std(), 0, 1, 2, 3, 4]}, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'bool_data': [False, False, True, True], 'str_data': ['a', 'b', 'c', 'a']})
    result = df.describe()
    expected = DataFrame({'bool_data': [4, 2, False, 2], 'str_data': [4, 3, 'a', 2]}, index=['count', 'unique', 'top', 'freq'])
    tm.assert_frame_equal(result, expected)