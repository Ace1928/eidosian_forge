from itertools import product
from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_cumcount_empty(self):
    ge = DataFrame().groupby(level=0)
    se = Series(dtype=object).groupby(level=0)
    e = Series(dtype='int64')
    tm.assert_series_equal(e, ge.cumcount())
    tm.assert_series_equal(e, se.cumcount())