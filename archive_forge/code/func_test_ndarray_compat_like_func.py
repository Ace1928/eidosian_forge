import inspect
import pydoc
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_ndarray_compat_like_func(self):
    s = Series(np.random.default_rng(2).standard_normal(10))
    result = Series(np.ones_like(s))
    expected = Series(1, index=range(10), dtype='float64')
    tm.assert_series_equal(result, expected)