from itertools import product
from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ngroup_mi(self):
    mi = MultiIndex.from_tuples([[0, 1], [1, 2], [2, 2], [2, 2], [1, 0]])
    df = DataFrame({'A': list('aaaba')}, index=mi)
    g = df.groupby('A')
    sg = g.A
    expected = Series([0, 0, 0, 1, 0], index=mi)
    tm.assert_series_equal(expected, g.ngroup())
    tm.assert_series_equal(expected, sg.ngroup())