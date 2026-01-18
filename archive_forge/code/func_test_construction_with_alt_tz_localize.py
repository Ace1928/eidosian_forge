from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
@pytest.mark.parametrize('kwargs', [{'tz': 'dtype.tz'}, {'dtype': 'dtype'}, {'dtype': 'dtype', 'tz': 'dtype.tz'}])
def test_construction_with_alt_tz_localize(self, kwargs, tz_aware_fixture):
    tz = tz_aware_fixture
    i = date_range('20130101', periods=5, freq='h', tz=tz)
    i = i._with_freq(None)
    kwargs = {key: attrgetter(val)(i) for key, val in kwargs.items()}
    if 'tz' in kwargs:
        result = DatetimeIndex(i.asi8, tz='UTC').tz_convert(kwargs['tz'])
        expected = DatetimeIndex(i, **kwargs)
        tm.assert_index_equal(result, expected)
    i2 = DatetimeIndex(i.tz_localize(None).asi8, tz='UTC')
    expected = i.tz_localize(None).tz_localize('UTC')
    tm.assert_index_equal(i2, expected)
    msg = 'cannot supply both a tz and a dtype with a tz'
    with pytest.raises(ValueError, match=msg):
        DatetimeIndex(i.tz_localize(None).asi8, dtype=i.dtype, tz='US/Pacific')