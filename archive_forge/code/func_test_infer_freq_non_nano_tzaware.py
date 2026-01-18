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
def test_infer_freq_non_nano_tzaware(tz_aware_fixture):
    tz = tz_aware_fixture
    dti = date_range('2016-01-01', periods=365, freq='B', tz=tz)
    dta = dti._data.as_unit('s')
    res = frequencies.infer_freq(dta)
    assert res == 'B'