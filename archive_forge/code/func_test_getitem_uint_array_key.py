from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
def test_getitem_uint_array_key(self, any_unsigned_int_numpy_dtype):
    ser = Series([1, 2, 3])
    key = np.array([4], dtype=any_unsigned_int_numpy_dtype)
    with pytest.raises(KeyError, match='4'):
        ser[key]
    with pytest.raises(KeyError, match='4'):
        ser.loc[key]