import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import Hour
def test_intersection_equal(self, sort):
    first = timedelta_range('1 day', periods=4, freq='h')
    second = timedelta_range('1 day', periods=4, freq='h')
    intersect = first.intersection(second, sort=sort)
    if sort is None:
        tm.assert_index_equal(intersect, second.sort_values())
    tm.assert_index_equal(intersect, second)
    inter = first.intersection(first, sort=sort)
    assert inter is first