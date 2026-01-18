from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_astype_to_td64d_raises(self, index_or_series):
    scalar = Timedelta(days=31)
    td = index_or_series([scalar, scalar, scalar + timedelta(minutes=5, seconds=3), NaT], dtype='m8[ns]')
    msg = "Cannot convert from timedelta64\\[ns\\] to timedelta64\\[D\\]. Supported resolutions are 's', 'ms', 'us', 'ns'"
    with pytest.raises(ValueError, match=msg):
        td.astype('timedelta64[D]')