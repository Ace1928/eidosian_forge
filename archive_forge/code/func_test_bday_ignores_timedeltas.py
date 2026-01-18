from __future__ import annotations
from datetime import (
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.offsets import (
from pandas import (
from pandas.tests.tseries.offsets.common import assert_offset_equal
@pytest.mark.parametrize('td_unit', ['s', 'ms', 'us', 'ns'])
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_bday_ignores_timedeltas(self, unit, td_unit):
    idx = date_range('2010/02/01', '2010/02/10', freq='12h', unit=unit)
    td = Timedelta(3, unit='h').as_unit(td_unit)
    off = BDay(offset=td)
    t1 = idx + off
    exp_unit = tm.get_finest_unit(td.unit, idx.unit)
    expected = DatetimeIndex(['2010-02-02 03:00:00', '2010-02-02 15:00:00', '2010-02-03 03:00:00', '2010-02-03 15:00:00', '2010-02-04 03:00:00', '2010-02-04 15:00:00', '2010-02-05 03:00:00', '2010-02-05 15:00:00', '2010-02-08 03:00:00', '2010-02-08 15:00:00', '2010-02-08 03:00:00', '2010-02-08 15:00:00', '2010-02-08 03:00:00', '2010-02-08 15:00:00', '2010-02-09 03:00:00', '2010-02-09 15:00:00', '2010-02-10 03:00:00', '2010-02-10 15:00:00', '2010-02-11 03:00:00'], freq=None).as_unit(exp_unit)
    tm.assert_index_equal(t1, expected)
    pointwise = DatetimeIndex([x + off for x in idx]).as_unit(exp_unit)
    tm.assert_index_equal(pointwise, expected)