from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_getitem_2d(self, arr1d):
    expected = type(arr1d)._simple_new(arr1d._ndarray[:, np.newaxis], dtype=arr1d.dtype)
    result = arr1d[:, np.newaxis]
    tm.assert_equal(result, expected)
    arr2d = expected
    expected = type(arr2d)._simple_new(arr2d._ndarray[:3, 0], dtype=arr2d.dtype)
    result = arr2d[:3, 0]
    tm.assert_equal(result, expected)
    result = arr2d[-1, 0]
    expected = arr1d[-1]
    assert result == expected