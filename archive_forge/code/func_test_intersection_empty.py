from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
@pytest.mark.parametrize('freq', ['min', 'B'])
def test_intersection_empty(self, tz_aware_fixture, freq):
    tz = tz_aware_fixture
    rng = date_range('6/1/2000', '6/15/2000', freq=freq, tz=tz)
    result = rng[0:0].intersection(rng)
    assert len(result) == 0
    assert result.freq == rng.freq
    result = rng.intersection(rng[0:0])
    assert len(result) == 0
    assert result.freq == rng.freq
    check_freq = freq != 'min'
    result = rng[:3].intersection(rng[-3:])
    tm.assert_index_equal(result, rng[:0])
    if check_freq:
        assert result.freq == rng.freq
    result = rng[-3:].intersection(rng[:3])
    tm.assert_index_equal(result, rng[:0])
    if check_freq:
        assert result.freq == rng.freq