from datetime import timedelta
import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas import (
def test_add_overflow_raises(self):
    per = Timestamp.max.to_period('ns')
    msg = '|'.join(['Python int too large to convert to C long', 'int too big to convert'])
    with pytest.raises(OverflowError, match=msg):
        per + 1
    msg = 'value too large'
    with pytest.raises(OverflowError, match=msg):
        per + Timedelta(1)
    with pytest.raises(OverflowError, match=msg):
        per + offsets.Nano(1)