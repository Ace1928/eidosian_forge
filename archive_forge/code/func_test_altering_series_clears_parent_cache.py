from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_altering_series_clears_parent_cache(self, using_copy_on_write, warn_copy_on_write):
    df = DataFrame([[1, 2], [3, 4]], index=['a', 'b'], columns=['A', 'B'])
    ser = df['A']
    if using_copy_on_write or warn_copy_on_write:
        assert 'A' not in df._item_cache
    else:
        assert 'A' in df._item_cache
    ser['c'] = 5
    assert len(ser) == 3
    assert 'A' not in df._item_cache
    assert df['A'] is not ser
    assert len(df['A']) == 2