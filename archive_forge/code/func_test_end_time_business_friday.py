import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
@pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
def test_end_time_business_friday(self):
    pi = period_range('1990-01-05', freq='B', periods=1)
    result = pi.end_time
    dti = date_range('1990-01-05', freq='D', periods=1)._with_freq(None)
    expected = dti + Timedelta(days=1, nanoseconds=-1)
    tm.assert_index_equal(result, expected)