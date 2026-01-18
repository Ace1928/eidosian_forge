import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_assignment_with_dups(self):
    cols = MultiIndex.from_tuples([('A', '1'), ('B', '1'), ('A', '2')])
    df = DataFrame(np.arange(3).reshape((1, 3)), columns=cols, dtype=object)
    index = df.index.copy()
    df['A'] = df['A'].astype(np.float64)
    tm.assert_index_equal(df.index, index)