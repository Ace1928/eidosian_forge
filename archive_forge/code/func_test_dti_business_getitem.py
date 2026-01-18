from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
@pytest.mark.parametrize('freq', ['B', 'C'])
def test_dti_business_getitem(self, freq):
    rng = bdate_range(START, END, freq=freq)
    smaller = rng[:5]
    exp = DatetimeIndex(rng.view(np.ndarray)[:5], freq=freq)
    tm.assert_index_equal(smaller, exp)
    assert smaller.freq == exp.freq
    assert smaller.freq == rng.freq
    sliced = rng[::5]
    assert sliced.freq == to_offset(freq) * 5
    fancy_indexed = rng[[4, 3, 2, 1, 0]]
    assert len(fancy_indexed) == 5
    assert isinstance(fancy_indexed, DatetimeIndex)
    assert fancy_indexed.freq is None
    assert rng[4] == rng[np_long(4)]