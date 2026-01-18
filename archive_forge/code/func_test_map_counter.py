from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_counter():
    s = Series(['a', 'b', 'c'], index=[1, 2, 3])
    counter = Counter()
    counter['b'] = 5
    counter['c'] += 1
    result = s.map(counter)
    expected = Series([0, 5, 1], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)