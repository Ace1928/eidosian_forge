import datetime
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
def test_itertuples_duplicate_cols(self):
    df = DataFrame(data={'a': [1, 2, 3], 'b': [4, 5, 6]})
    dfaa = df[['a', 'a']]
    assert list(dfaa.itertuples()) == [(0, 1, 1), (1, 2, 2), (2, 3, 3)]
    if not (is_platform_windows() or not IS64):
        assert repr(list(df.itertuples(name=None))) == '[(0, 1, 4), (1, 2, 5), (2, 3, 6)]'