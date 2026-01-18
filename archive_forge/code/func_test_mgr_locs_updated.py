import numpy as np
import pytest
from pandas._libs import lib
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('func', [cumsum_max, pytest.param(assert_block_lengths, marks=td.skip_array_manager_invalid_test)])
def test_mgr_locs_updated(func):
    df = pd.DataFrame({'A': ['a', 'a', 'a'], 'B': ['a', 'b', 'b'], 'C': [1, 1, 1]})
    result = df.groupby(['A', 'B']).agg(func)
    expected = pd.DataFrame({'C': [0, 0]}, index=pd.MultiIndex.from_product([['a'], ['a', 'b']], names=['A', 'B']))
    tm.assert_frame_equal(result, expected)