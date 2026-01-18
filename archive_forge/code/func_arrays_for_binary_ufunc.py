from collections import deque
import re
import string
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray
@pytest.fixture
def arrays_for_binary_ufunc():
    """
    A pair of random, length-100 integer-dtype arrays, that are mostly 0.
    """
    a1 = np.random.default_rng(2).integers(0, 10, 100, dtype='int64')
    a2 = np.random.default_rng(2).integers(0, 10, 100, dtype='int64')
    a1[::3] = 0
    a2[::4] = 0
    return (a1, a2)