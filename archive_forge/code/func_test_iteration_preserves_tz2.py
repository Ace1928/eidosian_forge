import dateutil.tz
import numpy as np
import pytest
from pandas import (
from pandas.core.arrays import datetimes
def test_iteration_preserves_tz2(self):
    index = date_range('2012-01-01', periods=3, freq='h', tz=dateutil.tz.tzoffset(None, -28800))
    for i, ts in enumerate(index):
        result = ts
        expected = index[i]
        assert result._repr_base == expected._repr_base
        assert result == expected