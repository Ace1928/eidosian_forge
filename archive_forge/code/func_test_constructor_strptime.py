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
def test_constructor_strptime(self):
    fmt = '%Y%m%d-%H%M%S-%f%z'
    ts = '20190129-235348-000001+0000'
    msg = 'Timestamp.strptime\\(\\) is not implemented'
    with pytest.raises(NotImplementedError, match=msg):
        Timestamp.strptime(ts, fmt)