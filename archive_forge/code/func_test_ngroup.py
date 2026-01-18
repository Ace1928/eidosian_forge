from itertools import product
from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ngroup(self):
    df = DataFrame({'A': list('aaaba')})
    g = df.groupby('A')
    sg = g.A
    expected = Series([0, 0, 0, 1, 0])
    tm.assert_series_equal(expected, g.ngroup())
    tm.assert_series_equal(expected, sg.ngroup())