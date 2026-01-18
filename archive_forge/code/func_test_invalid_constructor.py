from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('w', [2.0, 'foo', np.array([2])])
def test_invalid_constructor(frame_or_series, w):
    c = frame_or_series(range(5)).rolling
    msg = '|'.join(['window must be an integer', 'passed window foo is not compatible with a datetimelike index'])
    with pytest.raises(ValueError, match=msg):
        c(window=w)
    msg = 'min_periods must be an integer'
    with pytest.raises(ValueError, match=msg):
        c(window=2, min_periods=w)
    msg = 'center must be a boolean'
    with pytest.raises(ValueError, match=msg):
        c(window=2, min_periods=1, center=w)