from datetime import (
import numpy as np
import pytest
from pandas.errors import UnsortedIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
def test_loc_ax_single_level_indexer_simple_df(self):
    df = DataFrame(np.arange(9).reshape(3, 3), columns=['a', 'b', 'c'])
    result = df.loc(axis=1)['a']
    expected = Series(np.array([0, 3, 6]), name='a')
    tm.assert_series_equal(result, expected)