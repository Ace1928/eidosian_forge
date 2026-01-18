import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_assert_extension_array_equal_time_units():
    timestamp = Timestamp('2023-11-04T12')
    naive = array([timestamp], dtype='datetime64[ns]')
    utc = array([timestamp], dtype='datetime64[ns, UTC]')
    tm.assert_extension_array_equal(naive, utc, check_dtype=False)
    tm.assert_extension_array_equal(utc, naive, check_dtype=False)