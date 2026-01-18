import pytest
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsTimedelta
from pandas import Timedelta
def test_as_unit_non_nano(self):
    td = Timedelta(days=1).as_unit('ms')
    assert td.days == 1
    assert td._value == 86400000
    assert td.components.days == 1
    assert td._d == 1
    assert td.total_seconds() == 86400
    res = td.as_unit('us')
    assert res._value == 86400000000
    assert res.components.days == 1
    assert res.components.hours == 0
    assert res._d == 1
    assert res._h == 0
    assert res.total_seconds() == 86400