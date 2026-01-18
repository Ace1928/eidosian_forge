from __future__ import annotations
from datetime import (
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.offsets import (
from pandas import (
from pandas.tests.tseries.offsets.common import assert_offset_equal
def test_add_bday_offset_nanos(self):
    idx = date_range('2010/02/01', '2010/02/10', freq='12h', unit='ns')
    off = BDay(offset=Timedelta(3, unit='ns'))
    result = idx + off
    expected = DatetimeIndex([x + off for x in idx])
    tm.assert_index_equal(result, expected)