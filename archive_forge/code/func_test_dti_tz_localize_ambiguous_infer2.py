from datetime import (
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', easts)
def test_dti_tz_localize_ambiguous_infer2(self, tz, unit):
    dr = date_range(datetime(2011, 11, 6, 0), periods=5, freq=offsets.Hour(), tz=tz, unit=unit)
    times = ['11/06/2011 00:00', '11/06/2011 01:00', '11/06/2011 01:00', '11/06/2011 02:00', '11/06/2011 03:00']
    di = DatetimeIndex(times).as_unit(unit)
    result = di.tz_localize(tz, ambiguous='infer')
    expected = dr._with_freq(None)
    tm.assert_index_equal(result, expected)
    result2 = DatetimeIndex(times, tz=tz, ambiguous='infer').as_unit(unit)
    tm.assert_index_equal(result2, expected)