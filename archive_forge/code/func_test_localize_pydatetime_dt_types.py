from datetime import datetime
import numpy as np
import pytest
from pytz import UTC
from pandas._libs.tslibs import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dt, expected', [pytest.param(Timestamp('2000-01-01'), Timestamp('2000-01-01', tz=UTC), id='timestamp'), pytest.param(datetime(2000, 1, 1), datetime(2000, 1, 1, tzinfo=UTC), id='datetime'), pytest.param(SubDatetime(2000, 1, 1), SubDatetime(2000, 1, 1, tzinfo=UTC), id='subclassed_datetime')])
def test_localize_pydatetime_dt_types(dt, expected):
    result = conversion.localize_pydatetime(dt, UTC)
    assert result == expected