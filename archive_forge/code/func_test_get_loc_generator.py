import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
def test_get_loc_generator(self, index):
    exc = KeyError
    if isinstance(index, (DatetimeIndex, TimedeltaIndex, PeriodIndex, IntervalIndex, MultiIndex)):
        exc = InvalidIndexError
    with pytest.raises(exc, match='generator object'):
        index.get_loc((x for x in range(5)))