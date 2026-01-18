import calendar
from datetime import (
import zoneinfo
import dateutil.tz
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.compat import PY310
from pandas.errors import OutOfBoundsDatetime
from pandas import (
@pytest.mark.parametrize('z', ['Z0', 'Z00'])
def test_constructor_invalid_Z0_isostring(self, z):
    msg = f'Unknown datetime string format, unable to parse: 2014-11-02 01:00{z}'
    with pytest.raises(ValueError, match=msg):
        Timestamp(f'2014-11-02 01:00{z}')