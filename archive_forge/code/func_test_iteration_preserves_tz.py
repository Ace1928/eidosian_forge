import dateutil.tz
import numpy as np
import pytest
from pandas import (
from pandas.core.arrays import datetimes
def test_iteration_preserves_tz(self):
    index = date_range('2012-01-01', periods=3, freq='h', tz='US/Eastern')
    for i, ts in enumerate(index):
        result = ts
        expected = index[i]
        assert result == expected