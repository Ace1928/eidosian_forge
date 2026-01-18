from datetime import (
import re
import numpy as np
import pytest
import pytz
from pytz import timezone
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.offsets import (
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.datetimes import _generate_range as generate_range
from pandas.tests.indexes.datetimes.test_timezones import (
from pandas.tseries.holiday import USFederalHolidayCalendar
@pytest.mark.parametrize('freq,freq_depr', [('h', 'H'), ('2min', '2T'), ('1s', '1S'), ('2ms', '2L'), ('1us', '1U'), ('2ns', '2N')])
def test_frequencies_H_T_S_L_U_N_deprecated(self, freq, freq_depr):
    freq_msg = re.split('[0-9]*', freq, maxsplit=1)[1]
    freq_depr_msg = re.split('[0-9]*', freq_depr, maxsplit=1)[1]
    msg = f"'{freq_depr_msg}' is deprecated and will be removed in a future version, "
    f"please use '{freq_msg}' instead"
    expected = date_range('1/1/2000', periods=2, freq=freq)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = date_range('1/1/2000', periods=2, freq=freq_depr)
    tm.assert_index_equal(result, expected)