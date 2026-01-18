import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('downcast,expected_dtype', [('integer', np.int16), ('signed', np.int16), ('unsigned', np.uint16)])
def test_downcast_not8bit(downcast, expected_dtype):
    data = ['256', 257, 258]
    expected = np.array([256, 257, 258], dtype=expected_dtype)
    res = to_numeric(data, downcast=downcast)
    tm.assert_numpy_array_equal(res, expected)