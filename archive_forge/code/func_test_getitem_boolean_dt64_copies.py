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
def test_getitem_boolean_dt64_copies(self):
    dti = date_range('2016-01-01', periods=4, tz='US/Pacific')
    key = np.array([True, True, False, False])
    ser = Series(dti._data)
    res = ser[key]
    assert res._values._ndarray.base is None
    ser2 = Series(range(4))
    res2 = ser2[key]
    assert res2._values.base is None