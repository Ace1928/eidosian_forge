import numpy as np
import pytest
from pandas._libs.missing import is_matching_na
from pandas.core.dtypes.common import (
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE
def test_delete_2d(self, data):
    arr2d = data.repeat(3).reshape(-1, 3)
    result = arr2d.delete(1, axis=0)
    expected = data.delete(1).repeat(3).reshape(-1, 3)
    tm.assert_extension_array_equal(result, expected)
    result = arr2d.delete(1, axis=1)
    expected = data.repeat(2).reshape(-1, 2)
    tm.assert_extension_array_equal(result, expected)