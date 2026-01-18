import dateutil.tz
import numpy as np
import pytest
from pandas import (
from pandas.core.arrays import datetimes
def test_iteration_preserves_tz3(self):
    index = DatetimeIndex(['2014-12-01 03:32:39.987000-08:00', '2014-12-01 04:12:34.987000-08:00'])
    for i, ts in enumerate(index):
        result = ts
        expected = index[i]
        assert result._repr_base == expected._repr_base
        assert result == expected