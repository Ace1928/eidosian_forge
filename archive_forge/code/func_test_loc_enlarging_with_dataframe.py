import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_loc_enlarging_with_dataframe(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3]})
    rhs = DataFrame({'b': [1, 2, 3], 'c': [4, 5, 6]})
    rhs_orig = rhs.copy()
    df.loc[:, ['b', 'c']] = rhs
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, 'b'), get_array(rhs, 'b'))
        assert np.shares_memory(get_array(df, 'c'), get_array(rhs, 'c'))
        assert not df._mgr._has_no_reference(1)
    else:
        assert not np.shares_memory(get_array(df, 'b'), get_array(rhs, 'b'))
    df.iloc[0, 1] = 100
    tm.assert_frame_equal(rhs, rhs_orig)