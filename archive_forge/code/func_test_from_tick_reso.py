from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
def test_from_tick_reso():
    tick = offsets.Nano()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_ns.value
    tick = offsets.Micro()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_us.value
    tick = offsets.Milli()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_ms.value
    tick = offsets.Second()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_s.value
    tick = offsets.Minute()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_s.value
    tick = offsets.Hour()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_s.value
    tick = offsets.Day()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_s.value