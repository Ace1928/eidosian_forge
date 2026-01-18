import datetime as dt
from itertools import combinations
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_append_sorts(self, sort):
    df1 = DataFrame({'a': [1, 2], 'b': [1, 2]}, columns=['b', 'a'])
    df2 = DataFrame({'a': [1, 2], 'c': [3, 4]}, index=[2, 3])
    result = df1._append(df2, sort=sort)
    expected = DataFrame({'b': [1, 2, None, None], 'a': [1, 2, 1, 2], 'c': [None, None, 3, 4]}, columns=['a', 'b', 'c'])
    if sort is False:
        expected = expected[['b', 'a', 'c']]
    tm.assert_frame_equal(result, expected)