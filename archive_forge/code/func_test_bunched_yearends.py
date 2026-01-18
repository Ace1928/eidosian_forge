from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytest
from pandas import Timestamp
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
def test_bunched_yearends():
    fy = FY5253(n=1, weekday=5, startingMonth=12, variation='nearest')
    dt = Timestamp('2004-01-01')
    assert fy.rollback(dt) == Timestamp('2002-12-28')
    assert (-fy)._apply(dt) == Timestamp('2002-12-28')
    assert dt - fy == Timestamp('2002-12-28')
    assert fy.rollforward(dt) == Timestamp('2004-01-03')
    assert fy._apply(dt) == Timestamp('2004-01-03')
    assert fy + dt == Timestamp('2004-01-03')
    assert dt + fy == Timestamp('2004-01-03')
    dt = Timestamp('2003-12-31')
    assert fy.rollback(dt) == Timestamp('2002-12-28')
    assert (-fy)._apply(dt) == Timestamp('2002-12-28')
    assert dt - fy == Timestamp('2002-12-28')