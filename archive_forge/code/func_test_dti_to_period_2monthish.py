import dateutil.tz
from dateutil.tz import tzlocal
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import MONTHS
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq_offset, freq_period', [('2ME', '2M'), (MonthEnd(2), MonthEnd(2))])
def test_dti_to_period_2monthish(self, freq_offset, freq_period):
    dti = date_range('2020-01-01', periods=3, freq=freq_offset)
    pi = dti.to_period()
    tm.assert_index_equal(pi, period_range('2020-01', '2020-05', freq=freq_period))