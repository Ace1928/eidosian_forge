import pytest
from pandas import (
from pandas.tseries.offsets import (
@pytest.mark.parametrize('values', [['20180101', '20180103', '20180105'], []])
@pytest.mark.parametrize('freq', ['2D', Day(2), '2B', BDay(2), '48h', Hour(48)])
@pytest.mark.parametrize('tz', [None, 'US/Eastern'])
def test_freq_setter(self, values, freq, tz):
    idx = DatetimeIndex(values, tz=tz)
    idx._data.freq = freq
    assert idx.freq == freq
    assert isinstance(idx.freq, DateOffset)
    idx._data.freq = None
    assert idx.freq is None