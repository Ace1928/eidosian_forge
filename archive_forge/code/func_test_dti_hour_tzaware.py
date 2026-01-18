import calendar
from datetime import (
import locale
import unicodedata
import numpy as np
import pytest
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
@pytest.mark.parametrize('prefix', ['', 'dateutil/'])
def test_dti_hour_tzaware(self, prefix):
    strdates = ['1/1/2012', '3/1/2012', '4/1/2012']
    rng = DatetimeIndex(strdates, tz=prefix + 'US/Eastern')
    assert (rng.hour == 0).all()
    dr = date_range('2011-10-02 00:00', freq='h', periods=10, tz=prefix + 'America/Atikokan')
    expected = Index(np.arange(10, dtype=np.int32))
    tm.assert_index_equal(dr.hour, expected)