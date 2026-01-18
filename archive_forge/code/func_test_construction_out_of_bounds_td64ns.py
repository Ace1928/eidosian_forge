from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('val, unit', [(15251, 'W'), (106752, 'D'), (2562048, 'h'), (153722868, 'm'), (9223372037, 's')])
def test_construction_out_of_bounds_td64ns(val, unit):
    td64 = np.timedelta64(val, unit)
    assert td64.astype('m8[ns]').view('i8') < 0
    td = Timedelta(td64)
    if unit != 'M':
        assert td.asm8 == td64
    assert td.asm8.dtype == 'm8[s]'
    msg = "Cannot cast 1067\\d\\d days .* to unit='ns' without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        td.as_unit('ns')
    assert Timedelta(td64 - 1) == td64 - 1
    td64 *= -1
    assert td64.astype('m8[ns]').view('i8') > 0
    td2 = Timedelta(td64)
    msg = "Cannot cast -1067\\d\\d days .* to unit='ns' without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        td2.as_unit('ns')
    assert Timedelta(td64 + 1) == td64 + 1