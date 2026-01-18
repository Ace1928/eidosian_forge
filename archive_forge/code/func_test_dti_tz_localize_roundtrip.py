from datetime import (
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
def test_dti_tz_localize_roundtrip(self, tz_aware_fixture):
    idx = date_range(start='2014-06-01', end='2014-08-30', freq='15min')
    tz = tz_aware_fixture
    localized = idx.tz_localize(tz)
    with pytest.raises(TypeError, match='Already tz-aware, use tz_convert to convert'):
        localized.tz_localize(tz)
    reset = localized.tz_localize(None)
    assert reset.tzinfo is None
    expected = idx._with_freq(None)
    tm.assert_index_equal(reset, expected)