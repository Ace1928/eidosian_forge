from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_get_group_empty_bins(self, observed):
    d = DataFrame([3, 1, 7, 6])
    bins = [0, 5, 10, 15]
    g = d.groupby(pd.cut(d[0], bins), observed=observed)
    result = g.get_group(pd.Interval(0, 5))
    expected = DataFrame([3, 1], index=[0, 1])
    tm.assert_frame_equal(result, expected)
    msg = "Interval\\(10, 15, closed='right'\\)"
    with pytest.raises(KeyError, match=msg):
        g.get_group(pd.Interval(10, 15))