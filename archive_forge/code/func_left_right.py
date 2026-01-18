from collections import defaultdict
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
import pandas.core.common as com
from pandas.core.sorting import (
@pytest.fixture
def left_right():
    low, high, n = (-1 << 10, 1 << 10, 1 << 20)
    left = DataFrame(np.random.default_rng(2).integers(low, high, (n, 7)), columns=list('ABCDEFG'))
    left['left'] = left.sum(axis=1)
    i = np.random.default_rng(2).permutation(len(left))
    right = left.iloc[i].copy()
    right.columns = right.columns[:-1].tolist() + ['right']
    right.index = np.arange(len(right))
    right['right'] *= -1
    return (left, right)