from datetime import (
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('offset', [-1, 1])
def test_dti_tz_localize_nonexistent_shift_invalid(self, offset, warsaw):
    tz = warsaw
    dti = DatetimeIndex([Timestamp('2015-03-29 02:20:00')])
    msg = 'The provided timedelta will relocalize on a nonexistent time'
    with pytest.raises(ValueError, match=msg):
        dti.tz_localize(tz, nonexistent=timedelta(seconds=offset))