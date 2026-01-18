import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_corrwith_index_union(self):
    df1 = DataFrame(np.random.default_rng(2).random(size=(10, 2)), columns=['a', 'b'])
    df2 = DataFrame(np.random.default_rng(2).random(size=(10, 3)), columns=['a', 'b', 'c'])
    result = df1.corrwith(df2, drop=False).index.sort_values()
    expected = df1.columns.union(df2.columns).sort_values()
    tm.assert_index_equal(result, expected)