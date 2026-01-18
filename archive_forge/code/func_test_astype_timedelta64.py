from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_astype_timedelta64(self):
    idx = TimedeltaIndex([100000000000000.0, 'NaT', NaT, np.nan])
    msg = "Cannot convert from timedelta64\\[ns\\] to timedelta64. Supported resolutions are 's', 'ms', 'us', 'ns'"
    with pytest.raises(ValueError, match=msg):
        idx.astype('timedelta64')
    result = idx.astype('timedelta64[ns]')
    tm.assert_index_equal(result, idx)
    assert result is not idx
    result = idx.astype('timedelta64[ns]', copy=False)
    tm.assert_index_equal(result, idx)
    assert result is idx