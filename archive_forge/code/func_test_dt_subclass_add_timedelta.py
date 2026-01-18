from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas._testing as tm
@pytest.mark.parametrize('lh,rh', [(SubDatetime(2000, 1, 1), Timedelta(hours=1)), (Timedelta(hours=1), SubDatetime(2000, 1, 1))])
def test_dt_subclass_add_timedelta(lh, rh):
    result = lh + rh
    expected = SubDatetime(2000, 1, 1, 1)
    assert result == expected