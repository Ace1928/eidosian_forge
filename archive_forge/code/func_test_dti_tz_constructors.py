from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
@pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
def test_dti_tz_constructors(self, tzstr):
    """Test different DatetimeIndex constructions with timezone
        Follow-up of GH#4229
        """
    arr = ['11/10/2005 08:00:00', '11/10/2005 09:00:00']
    idx1 = to_datetime(arr).tz_localize(tzstr)
    idx2 = date_range(start='2005-11-10 08:00:00', freq='h', periods=2, tz=tzstr)
    idx2 = idx2._with_freq(None)
    idx3 = DatetimeIndex(arr, tz=tzstr)
    idx4 = DatetimeIndex(np.array(arr), tz=tzstr)
    for other in [idx2, idx3, idx4]:
        tm.assert_index_equal(idx1, other)