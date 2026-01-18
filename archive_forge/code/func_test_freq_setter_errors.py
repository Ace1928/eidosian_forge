import pytest
from pandas import (
from pandas.tseries.offsets import (
def test_freq_setter_errors(self):
    idx = DatetimeIndex(['20180101', '20180103', '20180105'])
    msg = 'Inferred frequency 2D from passed values does not conform to passed frequency 5D'
    with pytest.raises(ValueError, match=msg):
        idx._data.freq = '5D'
    with pytest.raises(ValueError, match='Invalid frequency'):
        idx._data.freq = 'foo'