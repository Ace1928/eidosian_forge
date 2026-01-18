import pytest
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import OutOfBoundsDatetime
from pandas import (
import pandas._testing as tm
def test_asfreq_MS(self):
    initial = Period('2013')
    assert initial.asfreq(freq='M', how='S') == Period('2013-01', 'M')
    msg = 'MS is not supported as period frequency'
    with pytest.raises(ValueError, match=msg):
        initial.asfreq(freq='MS', how='S')
    with pytest.raises(ValueError, match=msg):
        Period('2013-01', 'MS')