import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@td.skip_array_manager_not_yet_implemented
def test_multiindex_setitem(self):
    arrays = [np.array(['bar', 'bar', 'baz', 'qux', 'qux', 'bar']), np.array(['one', 'two', 'one', 'one', 'two', 'one']), np.arange(0, 6, 1)]
    df_orig = DataFrame(np.random.default_rng(2).standard_normal((6, 3)), index=arrays, columns=['A', 'B', 'C']).sort_index()
    expected = df_orig.loc[['bar']] * 2
    df = df_orig.copy()
    df.loc[['bar']] *= 2
    tm.assert_frame_equal(df.loc[['bar']], expected)
    msg = 'cannot align on a multi-index with out specifying the join levels'
    with pytest.raises(TypeError, match=msg):
        df.loc['bar'] *= 2