import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('cons', [Series, Index])
def test_dataframe_from_series_or_index_different_dtype(using_copy_on_write, cons):
    obj = cons([1, 2], dtype='int64')
    df = DataFrame(obj, dtype='int32')
    assert not np.shares_memory(get_array(obj), get_array(df, 0))
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)