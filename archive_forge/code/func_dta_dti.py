from __future__ import annotations
from datetime import timedelta
import operator
import numpy as np
import pytest
from pandas._libs.tslibs import tz_compare
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.fixture
def dta_dti(self, unit, dtype):
    tz = getattr(dtype, 'tz', None)
    dti = pd.date_range('2016-01-01', periods=55, freq='D', tz=tz)
    if tz is None:
        arr = np.asarray(dti).astype(f'M8[{unit}]')
    else:
        arr = np.asarray(dti.tz_convert('UTC').tz_localize(None)).astype(f'M8[{unit}]')
    dta = DatetimeArray._simple_new(arr, dtype=dtype)
    return (dta, dti)