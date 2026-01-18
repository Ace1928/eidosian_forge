from datetime import (
import dateutil.tz
import pytest
import pytz
from pandas._libs.tslibs import (
from pandas.compat import is_platform_windows
from pandas import Timestamp
@pytest.mark.parametrize('ordered', [True, False])
def test_infer_tz_mismatch(infer_setup, ordered):
    eastern, _, _, _, start_naive, end_naive = infer_setup
    msg = 'Inputs must both have the same timezone'
    utc = pytz.utc
    start = utc.localize(start_naive)
    end = conversion.localize_pydatetime(end_naive, eastern)
    args = (start, end) if ordered else (end, start)
    with pytest.raises(AssertionError, match=msg):
        timezones.infer_tzinfo(*args)