from datetime import (
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas import Timestamp
import pandas._testing as tm
def test_infer_all_nat(self):
    arr = np.array([NaT, np.nan], dtype=object)
    result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
    assert tz is None
    assert result.dtype == 'M8[s]'