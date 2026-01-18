import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_midx_read_only_bool_indexer():

    def mklbl(prefix, n):
        return [f'{prefix}{i}' for i in range(n)]
    idx = pd.MultiIndex.from_product([mklbl('A', 4), mklbl('B', 2), mklbl('C', 4), mklbl('D', 2)])
    cols = pd.MultiIndex.from_tuples([('a', 'foo'), ('a', 'bar'), ('b', 'foo'), ('b', 'bah')], names=['lvl0', 'lvl1'])
    df = DataFrame(1, index=idx, columns=cols).sort_index().sort_index(axis=1)
    mask = df['a', 'foo'] == 1
    expected_mask = mask.copy()
    result = df.loc[pd.IndexSlice[mask, :, ['C1', 'C3']], :]
    expected = df.loc[pd.IndexSlice[:, :, ['C1', 'C3']], :]
    tm.assert_frame_equal(result, expected)
    tm.assert_series_equal(mask, expected_mask)