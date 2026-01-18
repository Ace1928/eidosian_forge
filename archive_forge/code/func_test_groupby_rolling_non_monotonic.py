import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
def test_groupby_rolling_non_monotonic(self):
    shuffled = [3, 0, 1, 2]
    sec = 1000
    df = DataFrame([{'t': Timestamp(2 * x * sec), 'x': x + 1, 'c': 42} for x in shuffled])
    with pytest.raises(ValueError, match='.* must be monotonic'):
        df.groupby('c').rolling(on='t', window='3s')