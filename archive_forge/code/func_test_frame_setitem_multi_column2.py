import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_frame_setitem_multi_column2(self):
    columns = MultiIndex.from_tuples([('A', '1'), ('A', '2'), ('B', '1')])
    df = DataFrame(index=[1, 3, 5], columns=columns)
    df['A'] = 0.0
    assert (df['A'].values == 0).all()
    df['B', '1'] = [1, 2, 3]
    df['A'] = df['B', '1']
    sliced_a1 = df['A', '1']
    sliced_a2 = df['A', '2']
    sliced_b1 = df['B', '1']
    tm.assert_series_equal(sliced_a1, sliced_b1, check_names=False)
    tm.assert_series_equal(sliced_a2, sliced_b1, check_names=False)
    assert sliced_a1.name == ('A', '1')
    assert sliced_a2.name == ('A', '2')
    assert sliced_b1.name == ('B', '1')