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
def test_get_loc_nan(self, level, nulls_fixture):
    levels = [['a', 'b'], ['c', 'd']]
    key = ['b', 'd']
    levels[level] = np.array([0, nulls_fixture], dtype=type(nulls_fixture))
    key[level] = nulls_fixture
    idx = MultiIndex.from_product(levels)
    assert idx.get_loc(tuple(key)) == 3