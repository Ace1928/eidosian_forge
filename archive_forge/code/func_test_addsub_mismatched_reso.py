from datetime import timedelta
import sys
from hypothesis import (
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsTimedelta
from pandas import (
import pandas._testing as tm
def test_addsub_mismatched_reso(self, td):
    other = Timedelta(days=1).as_unit('us')
    result = td + other
    assert result._creso == other._creso
    assert result.days == td.days + 1
    result = other + td
    assert result._creso == other._creso
    assert result.days == td.days + 1
    result = td - other
    assert result._creso == other._creso
    assert result.days == td.days - 1
    result = other - td
    assert result._creso == other._creso
    assert result.days == 1 - td.days
    other2 = Timedelta(500)
    msg = "Cannot cast 106752 days 00:00:00 to unit='ns' without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        td + other2
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        other2 + td
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        td - other2
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        other2 - td