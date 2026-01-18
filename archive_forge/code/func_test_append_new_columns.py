import datetime as dt
from itertools import combinations
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_append_new_columns(self):
    df = DataFrame({'a': {'x': 1, 'y': 2}, 'b': {'x': 3, 'y': 4}})
    row = Series([5, 6, 7], index=['a', 'b', 'c'], name='z')
    expected = DataFrame({'a': {'x': 1, 'y': 2, 'z': 5}, 'b': {'x': 3, 'y': 4, 'z': 6}, 'c': {'z': 7}})
    result = df._append(row)
    tm.assert_frame_equal(result, expected)