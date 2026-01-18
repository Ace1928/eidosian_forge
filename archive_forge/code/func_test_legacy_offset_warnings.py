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
@pytest.mark.parametrize('offset_func', [_get_offset, lambda freq: date_range('2011-01-01', periods=5, freq=freq)])
@pytest.mark.parametrize('freq', ['WEEKDAY', 'EOM', 'W@MON', 'W@TUE', 'W@WED', 'W@THU', 'W@FRI', 'W@SAT', 'W@SUN', 'QE@JAN', 'QE@FEB', 'QE@MAR', 'YE@JAN', 'YE@FEB', 'YE@MAR', 'YE@APR', 'YE@MAY', 'YE@JUN', 'YE@JUL', 'YE@AUG', 'YE@SEP', 'YE@OCT', 'YE@NOV', 'YE@DEC', 'YE@JAN', 'WOM@1MON', 'WOM@2MON', 'WOM@3MON', 'WOM@4MON', 'WOM@1TUE', 'WOM@2TUE', 'WOM@3TUE', 'WOM@4TUE', 'WOM@1WED', 'WOM@2WED', 'WOM@3WED', 'WOM@4WED', 'WOM@1THU', 'WOM@2THU', 'WOM@3THU', 'WOM@4THU', 'WOM@1FRI', 'WOM@2FRI', 'WOM@3FRI', 'WOM@4FRI'])
def test_legacy_offset_warnings(offset_func, freq):
    with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
        offset_func(freq)