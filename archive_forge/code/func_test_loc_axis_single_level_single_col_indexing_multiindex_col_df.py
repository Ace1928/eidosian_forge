from datetime import (
import numpy as np
import pytest
from pandas.errors import UnsortedIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
def test_loc_axis_single_level_single_col_indexing_multiindex_col_df(self):
    df = DataFrame(np.arange(27).reshape(3, 9), columns=MultiIndex.from_product([['a1', 'a2', 'a3'], ['b1', 'b2', 'b3']]))
    result = df.loc(axis=1)['a1']
    expected = df.iloc[:, :3]
    expected.columns = ['b1', 'b2', 'b3']
    tm.assert_frame_equal(result, expected)