from itertools import product
from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ngroup_distinct(self):
    df = DataFrame({'A': list('abcde')})
    g = df.groupby('A')
    sg = g.A
    expected = Series(range(5), dtype='int64')
    tm.assert_series_equal(expected, g.ngroup())
    tm.assert_series_equal(expected, sg.ngroup())