import numpy as np
from pandas import (
from pandas.core.arrays import (
def test_empty_dt64(self):
    shape = (3, 9)
    result = DatetimeArray._empty(shape, dtype='datetime64[ns]')
    assert isinstance(result, DatetimeArray)
    assert result.shape == shape