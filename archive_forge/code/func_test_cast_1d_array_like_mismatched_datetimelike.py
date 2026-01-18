import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_arraylike_from_scalar
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
import pandas._testing as tm
def test_cast_1d_array_like_mismatched_datetimelike():
    td = np.timedelta64('NaT', 'ns')
    dt = np.datetime64('NaT', 'ns')
    with pytest.raises(TypeError, match='Cannot cast'):
        construct_1d_arraylike_from_scalar(td, 2, dt.dtype)
    with pytest.raises(TypeError, match='Cannot cast'):
        construct_1d_arraylike_from_scalar(np.timedelta64(4, 'ns'), 2, dt.dtype)
    with pytest.raises(TypeError, match='Cannot cast'):
        construct_1d_arraylike_from_scalar(dt, 2, td.dtype)
    with pytest.raises(TypeError, match='Cannot cast'):
        construct_1d_arraylike_from_scalar(np.datetime64(4, 'ns'), 2, td.dtype)