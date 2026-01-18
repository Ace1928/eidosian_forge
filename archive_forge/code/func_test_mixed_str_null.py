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
def test_mixed_str_null(nulls_fixture):
    values = np.array(['b', nulls_fixture, 'a', 'b'], dtype=object)
    result = safe_sort(values)
    expected = np.array(['a', 'b', 'b', nulls_fixture], dtype=object)
    tm.assert_numpy_array_equal(result, expected)