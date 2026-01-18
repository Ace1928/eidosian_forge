from itertools import product
from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ngroup_cumcount_pair(self):
    for p in product(range(3), repeat=4):
        df = DataFrame({'a': p})
        g = df.groupby(['a'])
        order = sorted(set(p))
        ngroupd = [order.index(val) for val in p]
        cumcounted = [p[:i].count(val) for i, val in enumerate(p)]
        tm.assert_series_equal(g.ngroup(), Series(ngroupd))
        tm.assert_series_equal(g.cumcount(), Series(cumcounted))