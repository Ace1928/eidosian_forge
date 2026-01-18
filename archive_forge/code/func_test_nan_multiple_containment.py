import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nan_multiple_containment(self):
    index_cls = Index
    idx = index_cls([1.0, np.nan])
    tm.assert_numpy_array_equal(idx.isin([1.0]), np.array([True, False]))
    tm.assert_numpy_array_equal(idx.isin([2.0, np.pi]), np.array([False, False]))
    tm.assert_numpy_array_equal(idx.isin([np.nan]), np.array([False, True]))
    tm.assert_numpy_array_equal(idx.isin([1.0, np.nan]), np.array([True, True]))
    idx = index_cls([1.0, 2.0])
    tm.assert_numpy_array_equal(idx.isin([np.nan]), np.array([False, False]))