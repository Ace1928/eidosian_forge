import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_index_to_frame(using_copy_on_write):
    idx = Index([1, 2, 3], name='a')
    expected = idx.copy(deep=True)
    df = idx.to_frame()
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, 'a'), idx._values)
        assert not df._mgr._has_no_reference(0)
    else:
        assert not np.shares_memory(get_array(df, 'a'), idx._values)
    df.iloc[0, 0] = 100
    tm.assert_index_equal(idx, expected)