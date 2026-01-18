from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
def test_from_td64_retain_resolution():
    obj = np.timedelta64(12345, 'ms')
    td = Timedelta(obj)
    assert td._value == obj.view('i8')
    assert td._creso == NpyDatetimeUnit.NPY_FR_ms.value
    obj2 = np.timedelta64(1234, 'D')
    td2 = Timedelta(obj2)
    assert td2._creso == NpyDatetimeUnit.NPY_FR_s.value
    assert td2 == obj2
    assert td2.days == 1234
    obj3 = np.timedelta64(1000000000000000000, 'us')
    td3 = Timedelta(obj3)
    assert td3.total_seconds() == 1000000000000
    assert td3._creso == NpyDatetimeUnit.NPY_FR_us.value