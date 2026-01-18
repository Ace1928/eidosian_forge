from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_min_periods1():
    df = DataFrame([0, 1, 2, 1, 0], columns=['a'])
    result = df['a'].rolling(3, center=True, min_periods=1).max()
    expected = Series([1.0, 2.0, 2.0, 2.0, 1.0], name='a')
    tm.assert_series_equal(result, expected)