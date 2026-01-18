from datetime import datetime
from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
@pytest.mark.parametrize('index', [Index([1, 2]), Index(['a', 'b']), Index([1.1, 2.2]), pd.MultiIndex.from_arrays([[1, 2], ['a', 'b']])])
def test_fails_on_no_datetime_index(index):
    name = type(index).__name__
    df = DataFrame({'a': range(len(index))}, index=index)
    msg = f"Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, but got an instance of '{name}'"
    with pytest.raises(TypeError, match=msg):
        df.groupby(Grouper(freq='D'))