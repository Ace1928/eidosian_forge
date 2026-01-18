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
def test_fifth_week_of_month_infer():
    index = DatetimeIndex(['2014-03-31', '2014-06-30', '2015-03-30'])
    assert frequencies.infer_freq(index) is None