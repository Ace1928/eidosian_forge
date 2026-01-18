from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_int():
    left = Series({'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 4})
    right = Series({1: 11, 2: 22, 3: 33})
    assert left.dtype == np.float64
    assert issubclass(right.dtype.type, np.integer)
    merged = left.map(right)
    assert merged.dtype == np.float64
    assert isna(merged['d'])
    assert not isna(merged['c'])