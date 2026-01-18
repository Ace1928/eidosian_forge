import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_getitem_tuple_plus_slice():
    df = DataFrame({'a': np.arange(10), 'b': np.arange(10), 'c': np.random.default_rng(2).standard_normal(10), 'd': np.random.default_rng(2).standard_normal(10)}).set_index(['a', 'b'])
    expected = df.loc[0, 0]
    result = df.loc[(0, 0), :]
    tm.assert_series_equal(result, expected)