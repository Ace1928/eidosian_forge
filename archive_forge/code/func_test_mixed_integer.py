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
def test_mixed_integer(self):
    values = np.array(['b', 1, 0, 'a', 0, 'b'], dtype=object)
    result = safe_sort(values)
    expected = np.array([0, 0, 1, 'a', 'b', 'b'], dtype=object)
    tm.assert_numpy_array_equal(result, expected)