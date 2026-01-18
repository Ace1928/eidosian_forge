from datetime import (
import itertools
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_multiindex_at_get(self):
    df = DataFrame({'a': [1, 2]}, index=[[1, 2], [3, 4]])
    assert df.index.nlevels == 2
    assert df.at[(1, 3), 'a'] == 1
    assert df.loc[(1, 3), 'a'] == 1
    series = df['a']
    assert series.index.nlevels == 2
    assert series.at[1, 3] == 1
    assert series.loc[1, 3] == 1