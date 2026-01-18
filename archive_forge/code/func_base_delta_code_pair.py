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
@pytest.fixture(params=[(timedelta(1), 'D'), (timedelta(hours=1), 'h'), (timedelta(minutes=1), 'min'), (timedelta(seconds=1), 's'), (np.timedelta64(1, 'ns'), 'ns'), (timedelta(microseconds=1), 'us'), (timedelta(microseconds=1000), 'ms')])
def base_delta_code_pair(request):
    return request.param