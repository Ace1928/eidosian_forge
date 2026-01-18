from itertools import product
from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ngroup_descending(self):
    df = DataFrame(['a', 'a', 'b', 'a', 'b'], columns=['A'])
    g = df.groupby(['A'])
    ascending = Series([0, 0, 1, 0, 1])
    descending = Series([1, 1, 0, 1, 0])
    tm.assert_series_equal(descending, g.ngroups - 1 - ascending)
    tm.assert_series_equal(ascending, g.ngroup(ascending=True))
    tm.assert_series_equal(descending, g.ngroup(ascending=False))