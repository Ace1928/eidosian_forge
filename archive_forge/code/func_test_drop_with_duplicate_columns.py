import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_drop_with_duplicate_columns(self):
    df = DataFrame([[1, 5, 7.0], [1, 5, 7.0], [1, 5, 7.0]], columns=['bar', 'a', 'a'])
    result = df.drop(['a'], axis=1)
    expected = DataFrame([[1], [1], [1]], columns=['bar'])
    tm.assert_frame_equal(result, expected)
    result = df.drop('a', axis=1)
    tm.assert_frame_equal(result, expected)