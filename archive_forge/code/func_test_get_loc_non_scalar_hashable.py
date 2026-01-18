import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
def test_get_loc_non_scalar_hashable(self, index):
    from enum import Enum

    class E(Enum):
        X1 = 'x1'
    assert not is_scalar(E.X1)
    exc = KeyError
    msg = "<E.X1: 'x1'>"
    if isinstance(index, (DatetimeIndex, TimedeltaIndex, PeriodIndex, IntervalIndex)):
        exc = InvalidIndexError
        msg = 'E.X1'
    with pytest.raises(exc, match=msg):
        index.get_loc(E.X1)