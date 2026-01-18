from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_intersection2(self):
    first = date_range('2020-01-01', periods=10)
    second = first[5:]
    intersect = first.intersection(second)
    tm.assert_index_equal(intersect, second)
    cases = [klass(second.values) for klass in [np.array, Series, list]]
    for case in cases:
        result = first.intersection(case)
        tm.assert_index_equal(result, second)
    third = Index(['a', 'b', 'c'])
    result = first.intersection(third)
    expected = Index([], dtype=object)
    tm.assert_index_equal(result, expected)