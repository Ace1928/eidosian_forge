from contextlib import nullcontext
import copy
import numpy as np
import pytest
from pandas._libs.missing import is_matching_na
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import is_float
from pandas import (
import pandas._testing as tm
def test_equals_none_vs_nan():
    ser = Series([1, None], dtype=object)
    ser2 = Series([1, np.nan], dtype=object)
    assert ser.equals(ser2)
    assert Index(ser, dtype=ser.dtype).equals(Index(ser2, dtype=ser2.dtype))
    assert ser.array.equals(ser2.array)