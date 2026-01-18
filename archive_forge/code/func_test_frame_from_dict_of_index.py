import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_frame_from_dict_of_index(using_copy_on_write):
    idx = Index([1, 2, 3])
    expected = idx.copy(deep=True)
    df = DataFrame({'a': idx}, copy=False)
    assert np.shares_memory(get_array(df, 'a'), idx._values)
    if using_copy_on_write:
        assert not df._mgr._has_no_reference(0)
        df.iloc[0, 0] = 100
        tm.assert_index_equal(idx, expected)