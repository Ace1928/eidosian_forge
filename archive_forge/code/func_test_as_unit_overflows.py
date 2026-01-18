import pytest
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsTimedelta
from pandas import Timedelta
def test_as_unit_overflows(self):
    us = 9223372800000000
    td = Timedelta._from_value_and_reso(us, NpyDatetimeUnit.NPY_FR_us.value)
    msg = "Cannot cast 106752 days 00:00:00 to unit='ns' without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        td.as_unit('ns')
    res = td.as_unit('ms')
    assert res._value == us // 1000
    assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value