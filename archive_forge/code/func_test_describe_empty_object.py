import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_describe_empty_object(self):
    df = DataFrame({'A': [None, None]}, dtype=object)
    result = df.describe()
    expected = DataFrame({'A': [0, 0, np.nan, np.nan]}, dtype=object, index=['count', 'unique', 'top', 'freq'])
    tm.assert_frame_equal(result, expected)
    result = df.iloc[:0].describe()
    tm.assert_frame_equal(result, expected)