from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_groupby_levels_and_columns(self):
    idx_names = ['x', 'y']
    idx = MultiIndex.from_tuples([(1, 1), (1, 2), (3, 4), (5, 6)], names=idx_names)
    df = DataFrame(np.arange(12).reshape(-1, 3), index=idx)
    by_levels = df.groupby(level=idx_names).mean()
    by_columns = df.reset_index().groupby(idx_names).mean()
    by_columns.columns = by_columns.columns.astype(np.int64)
    tm.assert_frame_equal(by_levels, by_columns)