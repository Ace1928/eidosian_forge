from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
@pytest.mark.parametrize('setop', ['union', 'intersection', 'symmetric_difference'])
def test_dti_setop_aware(self, setop):
    rng = date_range('2012-11-15 00:00:00', periods=6, freq='h', tz='US/Central')
    rng2 = date_range('2012-11-15 12:00:00', periods=6, freq='h', tz='US/Eastern')
    result = getattr(rng, setop)(rng2)
    left = rng.tz_convert('UTC')
    right = rng2.tz_convert('UTC')
    expected = getattr(left, setop)(right)
    tm.assert_index_equal(result, expected)
    assert result.tz == left.tz
    if len(result):
        assert result[0].tz is timezone.utc
        assert result[-1].tz is timezone.utc