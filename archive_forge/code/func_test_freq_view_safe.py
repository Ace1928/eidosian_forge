import pytest
from pandas import (
from pandas.tseries.offsets import (
def test_freq_view_safe(self):
    dti = date_range('2016-01-01', periods=5)
    dta = dti._data
    dti2 = DatetimeIndex(dta)._with_freq(None)
    assert dti2.freq is None
    assert dti.freq == 'D'
    assert dta.freq == 'D'