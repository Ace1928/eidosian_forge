from datetime import datetime
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_append_overlapping_interval_levels():
    ivl1 = pd.IntervalIndex.from_breaks([0.0, 1.0, 2.0])
    ivl2 = pd.IntervalIndex.from_breaks([0.5, 1.5, 2.5])
    mi1 = MultiIndex.from_product([ivl1, ivl1])
    mi2 = MultiIndex.from_product([ivl2, ivl2])
    result = mi1.append(mi2)
    expected = MultiIndex.from_tuples([(pd.Interval(0.0, 1.0), pd.Interval(0.0, 1.0)), (pd.Interval(0.0, 1.0), pd.Interval(1.0, 2.0)), (pd.Interval(1.0, 2.0), pd.Interval(0.0, 1.0)), (pd.Interval(1.0, 2.0), pd.Interval(1.0, 2.0)), (pd.Interval(0.5, 1.5), pd.Interval(0.5, 1.5)), (pd.Interval(0.5, 1.5), pd.Interval(1.5, 2.5)), (pd.Interval(1.5, 2.5), pd.Interval(0.5, 1.5)), (pd.Interval(1.5, 2.5), pd.Interval(1.5, 2.5))])
    tm.assert_index_equal(result, expected)