from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('level', [0, 1])
@pytest.mark.parametrize('dtypes', [[int, float], [float, int]])
def test_get_loc_implicit_cast(self, level, dtypes):
    levels = [['a', 'b'], ['c', 'd']]
    key = ['b', 'd']
    lev_dtype, key_dtype = dtypes
    levels[level] = np.array([0, 1], dtype=lev_dtype)
    key[level] = key_dtype(1)
    idx = MultiIndex.from_product(levels)
    assert idx.get_loc(tuple(key)) == 3