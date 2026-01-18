import re
import numpy as np
import pytest
from pandas._libs.tslibs.timedeltas import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('unit', ['s', 'ms', 'us'])
def test_ints_to_pytimedelta(unit):
    arr = np.arange(6, dtype=np.int64).view(f'm8[{unit}]')
    res = ints_to_pytimedelta(arr, box=False)
    expected = arr.astype(object)
    tm.assert_numpy_array_equal(res, expected)
    res = ints_to_pytimedelta(arr, box=True)
    expected = np.array([Timedelta(x) for x in arr], dtype=object)
    tm.assert_numpy_array_equal(res, expected)