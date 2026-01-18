from __future__ import annotations
import contextlib
import string
import warnings
import numpy as np
import pandas as pd
from packaging.version import Version
import pandas.testing as tm
def assert_numpy_array_equal(left, right):
    left_na = pd.isna(left)
    right_na = pd.isna(right)
    np.testing.assert_array_equal(left_na, right_na)
    left_valid = left[~left_na]
    right_valid = right[~right_na]
    np.testing.assert_array_equal(left_valid, right_valid)