from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.offsets import _get_offset
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.compat import is_platform_windows
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.tools.datetimes import to_datetime
from pandas.tseries import (
def test_infer_freq_non_nano():
    arr = np.arange(10).astype(np.int64).view('M8[s]')
    dta = DatetimeArray._simple_new(arr, dtype=arr.dtype)
    res = frequencies.infer_freq(dta)
    assert res == 's'
    arr2 = arr.view('m8[ms]')
    tda = TimedeltaArray._simple_new(arr2, dtype=arr2.dtype)
    res2 = frequencies.infer_freq(tda)
    assert res2 == 'ms'