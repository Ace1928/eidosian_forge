import datetime as dt
from itertools import combinations
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_append_records(self):
    arr1 = np.zeros((2,), dtype='i4,f4,S10')
    arr1[:] = [(1, 2.0, 'Hello'), (2, 3.0, 'World')]
    arr2 = np.zeros((3,), dtype='i4,f4,S10')
    arr2[:] = [(3, 4.0, 'foo'), (5, 6.0, 'bar'), (7.0, 8.0, 'baz')]
    df1 = DataFrame(arr1)
    df2 = DataFrame(arr2)
    result = df1._append(df2, ignore_index=True)
    expected = DataFrame(np.concatenate((arr1, arr2)))
    tm.assert_frame_equal(result, expected)