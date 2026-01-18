from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('func,expected', [('transform', Series(name=2, dtype=np.float64)), ('agg', Series(name=2, dtype=np.float64, index=Index([], dtype=np.float64, name=1))), ('apply', Series(name=2, dtype=np.float64, index=Index([], dtype=np.float64, name=1)))])
def test_evaluate_with_empty_groups(self, func, expected):
    df = DataFrame({1: [], 2: []})
    g = df.groupby(1, group_keys=False)
    result = getattr(g[2], func)(lambda x: x)
    tm.assert_series_equal(result, expected)