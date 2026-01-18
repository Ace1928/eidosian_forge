from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.strptime import array_strptime
from pandas import (
import pandas._testing as tm
def test_array_strptime_resolution_all_nat(self):
    arr = np.array([NaT, np.nan], dtype=object)
    fmt = '%Y-%m-%d %H:%M:%S'
    res, _ = array_strptime(arr, fmt=fmt, utc=False, creso=creso_infer)
    assert res.dtype == 'M8[s]'
    res, _ = array_strptime(arr, fmt=fmt, utc=True, creso=creso_infer)
    assert res.dtype == 'M8[s]'