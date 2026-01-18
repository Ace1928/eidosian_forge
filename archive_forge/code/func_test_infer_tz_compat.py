from datetime import (
import dateutil.tz
import pytest
import pytz
from pandas._libs.tslibs import (
from pandas.compat import is_platform_windows
from pandas import Timestamp
def test_infer_tz_compat(infer_setup):
    eastern, _, start, end, start_naive, end_naive = infer_setup
    assert timezones.infer_tzinfo(start, end) is conversion.localize_pydatetime(start_naive, eastern).tzinfo
    assert timezones.infer_tzinfo(start, None) is conversion.localize_pydatetime(start_naive, eastern).tzinfo
    assert timezones.infer_tzinfo(None, end) is conversion.localize_pydatetime(end_naive, eastern).tzinfo