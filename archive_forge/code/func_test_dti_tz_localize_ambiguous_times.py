from datetime import (
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', easts)
def test_dti_tz_localize_ambiguous_times(self, tz):
    dr = date_range(datetime(2011, 3, 13, 1, 30), periods=3, freq=offsets.Hour())
    with pytest.raises(pytz.NonExistentTimeError, match='2011-03-13 02:30:00'):
        dr.tz_localize(tz)
    dr = date_range(datetime(2011, 3, 13, 3, 30), periods=3, freq=offsets.Hour(), tz=tz)
    dr = date_range(datetime(2011, 11, 6, 1, 30), periods=3, freq=offsets.Hour())
    with pytest.raises(pytz.AmbiguousTimeError, match='Cannot infer dst time'):
        dr.tz_localize(tz)
    dr = date_range(datetime(2011, 3, 13), periods=48, freq=offsets.Minute(30), tz=pytz.utc)