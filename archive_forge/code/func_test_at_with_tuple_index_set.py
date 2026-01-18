from datetime import (
import itertools
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore:Setting a value on a view:FutureWarning')
def test_at_with_tuple_index_set():
    df = DataFrame({'a': [1, 2]}, index=[(1, 2), (3, 4)])
    assert df.index.nlevels == 1
    df.at[(1, 2), 'a'] = 2
    assert df.at[(1, 2), 'a'] == 2
    series = df['a']
    assert series.index.nlevels == 1
    series.at[1, 2] = 3
    assert series.at[1, 2] == 3