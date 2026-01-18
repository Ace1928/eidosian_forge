import numpy as np
import pytest
from pandas.compat import PY311
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def assert_matching(actual, expected, check_dtype=False):
    assert len(actual) == len(expected)
    for act, exp in zip(actual, expected):
        act = np.asarray(act)
        exp = np.asarray(exp)
        tm.assert_numpy_array_equal(act, exp, check_dtype=check_dtype)