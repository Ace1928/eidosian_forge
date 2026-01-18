from itertools import permutations
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_is_non_overlapping_monotonic(self, closed):
    tpls = [(0, 1), (2, 3), (4, 5), (6, 7)]
    idx = IntervalIndex.from_tuples(tpls, closed=closed)
    assert idx.is_non_overlapping_monotonic is True
    idx = IntervalIndex.from_tuples(tpls[::-1], closed=closed)
    assert idx.is_non_overlapping_monotonic is True
    tpls = [(0, 2), (1, 3), (4, 5), (6, 7)]
    idx = IntervalIndex.from_tuples(tpls, closed=closed)
    assert idx.is_non_overlapping_monotonic is False
    idx = IntervalIndex.from_tuples(tpls[::-1], closed=closed)
    assert idx.is_non_overlapping_monotonic is False
    tpls = [(0, 1), (2, 3), (6, 7), (4, 5)]
    idx = IntervalIndex.from_tuples(tpls, closed=closed)
    assert idx.is_non_overlapping_monotonic is False
    idx = IntervalIndex.from_tuples(tpls[::-1], closed=closed)
    assert idx.is_non_overlapping_monotonic is False
    if closed == 'both':
        idx = IntervalIndex.from_breaks(range(4), closed=closed)
        assert idx.is_non_overlapping_monotonic is False
    else:
        idx = IntervalIndex.from_breaks(range(4), closed=closed)
        assert idx.is_non_overlapping_monotonic is True