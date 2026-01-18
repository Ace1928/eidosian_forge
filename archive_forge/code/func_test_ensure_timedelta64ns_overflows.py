from datetime import datetime
import numpy as np
import pytest
from pytz import UTC
from pandas._libs.tslibs import (
from pandas import (
import pandas._testing as tm
def test_ensure_timedelta64ns_overflows():
    arr = np.arange(10).astype('m8[Y]') * 100
    msg = 'Cannot convert 300 years to timedelta64\\[ns\\] without overflow'
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        astype_overflowsafe(arr, dtype=np.dtype('m8[ns]'))