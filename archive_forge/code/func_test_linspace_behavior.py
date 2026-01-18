import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
@pytest.mark.parametrize('periods, freq', [(3, '2D'), (5, 'D'), (6, '19h12min'), (7, '16h'), (9, '12h')])
def test_linspace_behavior(self, periods, freq):
    result = timedelta_range(start='0 days', end='4 days', periods=periods)
    expected = timedelta_range(start='0 days', end='4 days', freq=freq)
    tm.assert_index_equal(result, expected)