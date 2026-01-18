from datetime import (
import dateutil.tz
import pytest
import pytz
from pandas._libs.tslibs import (
from pandas.compat import is_platform_windows
from pandas import Timestamp
def test_maybe_get_tz_invalid_types():
    with pytest.raises(TypeError, match="<class 'float'>"):
        timezones.maybe_get_tz(44.0)
    with pytest.raises(TypeError, match="<class 'module'>"):
        timezones.maybe_get_tz(pytz)
    msg = "<class 'pandas._libs.tslibs.timestamps.Timestamp'>"
    with pytest.raises(TypeError, match=msg):
        timezones.maybe_get_tz(Timestamp('2021-01-01', tz='UTC'))