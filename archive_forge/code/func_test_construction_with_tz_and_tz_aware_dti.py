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
def test_construction_with_tz_and_tz_aware_dti(self):
    dti = date_range('2016-01-01', periods=3, tz='US/Central')
    msg = 'data is already tz-aware US/Central, unable to set specified tz'
    with pytest.raises(TypeError, match=msg):
        DatetimeIndex(dti, tz='Asia/Tokyo')