import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@td.skip_array_manager_invalid_test
def test_cache_updating(using_copy_on_write, warn_copy_on_write):
    a = np.random.default_rng(2).random((10, 3))
    df = DataFrame(a, columns=['x', 'y', 'z'])
    df_original = df.copy()
    tuples = [(i, j) for i in range(5) for j in range(2)]
    index = MultiIndex.from_tuples(tuples)
    df.index = index
    with tm.raises_chained_assignment_error():
        df.loc[0]['z'].iloc[0] = 1.0
    if using_copy_on_write:
        assert df.loc[(0, 0), 'z'] == df_original.loc[0, 'z']
    else:
        result = df.loc[(0, 0), 'z']
        assert result == 1
    df.loc[(0, 0), 'z'] = 2
    result = df.loc[(0, 0), 'z']
    assert result == 2