from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError
from pandas.core.dtypes.common import is_list_like
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_dt64tz_setitem_does_not_mutate_dti(self, using_copy_on_write):
    dti = date_range('2016-01-01', periods=10, tz='US/Pacific')
    ts = dti[0]
    ser = Series(dti)
    assert ser._values is not dti
    if using_copy_on_write:
        assert ser._values._ndarray.base is dti._data._ndarray.base
        assert ser._mgr.arrays[0]._ndarray.base is dti._data._ndarray.base
    else:
        assert ser._values._ndarray.base is not dti._data._ndarray.base
        assert ser._mgr.arrays[0]._ndarray.base is not dti._data._ndarray.base
    assert ser._mgr.arrays[0] is not dti
    ser[::3] = NaT
    assert ser[0] is NaT
    assert dti[0] == ts