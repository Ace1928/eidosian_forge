from copy import (
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import (
import pandas._testing as tm
def test_frame_or_series_compound_dtypes(self, frame_or_series):

    def f(dtype):
        return construct(frame_or_series, shape=3, value=1, dtype=dtype)
    msg = f'compound dtypes are not implemented in the {frame_or_series.__name__} constructor'
    with pytest.raises(NotImplementedError, match=msg):
        f([('A', 'datetime64[h]'), ('B', 'str'), ('C', 'int32')])
    f('int64')
    f('float64')
    f('M8[ns]')