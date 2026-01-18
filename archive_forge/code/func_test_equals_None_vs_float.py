from contextlib import nullcontext
import copy
import numpy as np
import pytest
from pandas._libs.missing import is_matching_na
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import is_float
from pandas import (
import pandas._testing as tm
def test_equals_None_vs_float():
    left = Series([-np.inf, np.nan, -1.0, 0.0, 1.0, 10 / 3, np.inf], dtype=object)
    right = Series([None] * len(left))
    assert not left.equals(right)
    assert not right.equals(left)
    assert not left.to_frame().equals(right.to_frame())
    assert not right.to_frame().equals(left.to_frame())
    assert not Index(left, dtype='object').equals(Index(right, dtype='object'))
    assert not Index(right, dtype='object').equals(Index(left, dtype='object'))