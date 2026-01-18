import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_fillna_series_empty_arg_inplace(using_copy_on_write):
    ser = Series([1, np.nan, 2])
    arr = get_array(ser)
    ser.fillna({}, inplace=True)
    assert np.shares_memory(get_array(ser), arr)
    if using_copy_on_write:
        assert ser._mgr._has_no_reference(0)