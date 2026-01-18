from pandas import (
import pandas._testing as tm
def test_delete_doesnt_infer_freq(self):
    tdi = TimedeltaIndex(['1 Day', '2 Days', None, '3 Days', '4 Days'])
    result = tdi.delete(2)
    assert result.freq is None