from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_astype_from_float_to_str(self, dtype):
    ser = Series([0.1], dtype=dtype)
    result = ser.astype(str)
    expected = Series(['0.1'], dtype=object)
    tm.assert_series_equal(result, expected)