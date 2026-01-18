import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('indexer, exp_value', [(slice(None), 1.0), ((1, 2), np.nan)])
def test_multiindex_setitem_columns_enlarging(self, indexer, exp_value):
    mi = MultiIndex.from_tuples([(1, 2), (3, 4)])
    df = DataFrame([[1, 2], [3, 4]], index=mi, columns=['a', 'b'])
    df.loc[indexer, ['c', 'd']] = 1.0
    expected = DataFrame([[1, 2, 1.0, 1.0], [3, 4, exp_value, exp_value]], index=mi, columns=['a', 'b', 'c', 'd'])
    tm.assert_frame_equal(df, expected)