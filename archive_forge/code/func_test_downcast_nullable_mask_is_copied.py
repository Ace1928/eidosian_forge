import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_downcast_nullable_mask_is_copied():
    arr = pd.array([1, 2, pd.NA], dtype='Int64')
    result = to_numeric(arr, downcast='integer')
    expected = pd.array([1, 2, pd.NA], dtype='Int8')
    tm.assert_extension_array_equal(result, expected)
    arr[1] = pd.NA
    tm.assert_extension_array_equal(result, expected)