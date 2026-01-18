import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_dataframe_from_series_infer_datetime(using_copy_on_write):
    ser = Series([Timestamp('2019-12-31'), Timestamp('2020-12-31')], dtype=object)
    with tm.assert_produces_warning(FutureWarning, match='Dtype inference'):
        df = DataFrame(ser)
    assert not np.shares_memory(get_array(ser), get_array(df, 0))
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)