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
@pytest.mark.parametrize('typ', [int, float])
def test_construct_from_int_float_with_unit_out_of_bound_raises(self, typ):
    val = typ(150000000000000)
    msg = f"cannot convert input {val} with the unit 'D'"
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        Timestamp(val, unit='D')