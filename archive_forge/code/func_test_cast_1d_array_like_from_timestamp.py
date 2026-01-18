import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_arraylike_from_scalar
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
import pandas._testing as tm
def test_cast_1d_array_like_from_timestamp(fixed_now_ts):
    ts = fixed_now_ts + Timedelta(1)
    res = construct_1d_arraylike_from_scalar(ts, 2, np.dtype('M8[ns]'))
    assert res[0] == ts