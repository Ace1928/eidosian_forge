from datetime import (
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', easts)
def test_dti_tz_localize_ambiguous_infer(self, tz):
    dr = date_range(datetime(2011, 11, 6, 0), periods=5, freq=offsets.Hour())
    with pytest.raises(pytz.AmbiguousTimeError, match='Cannot infer dst time'):
        dr.tz_localize(tz)