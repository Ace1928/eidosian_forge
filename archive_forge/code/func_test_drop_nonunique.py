import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_drop_nonunique(self):
    df = DataFrame([['x-a', 'x', 'a', 1.5], ['x-a', 'x', 'a', 1.2], ['z-c', 'z', 'c', 3.1], ['x-a', 'x', 'a', 4.1], ['x-b', 'x', 'b', 5.1], ['x-b', 'x', 'b', 4.1], ['x-b', 'x', 'b', 2.2], ['y-a', 'y', 'a', 1.2], ['z-b', 'z', 'b', 2.1]], columns=['var1', 'var2', 'var3', 'var4'])
    grp_size = df.groupby('var1').size()
    drop_idx = grp_size.loc[grp_size == 1]
    idf = df.set_index(['var1', 'var2', 'var3'])
    result = idf.drop(drop_idx.index, level=0).reset_index()
    expected = df[-df.var1.isin(drop_idx.index)]
    result.index = expected.index
    tm.assert_frame_equal(result, expected)