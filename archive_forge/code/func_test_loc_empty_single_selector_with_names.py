import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_empty_single_selector_with_names():
    idx = MultiIndex.from_product([['a', 'b'], ['A', 'B']], names=[1, 0])
    s2 = Series(index=idx, dtype=np.float64)
    result = s2.loc['a']
    expected = Series([np.nan, np.nan], index=Index(['A', 'B'], name=0))
    tm.assert_series_equal(result, expected)