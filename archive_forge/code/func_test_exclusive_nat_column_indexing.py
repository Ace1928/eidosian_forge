import numpy as np
import pytest
import pandas._libs.index as libindex
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
def test_exclusive_nat_column_indexing(self):
    df = DataFrame({'a': [pd.NaT, pd.NaT, pd.NaT, pd.NaT], 'b': ['C1', 'C2', 'C3', 'C4'], 'c': [10, 15, np.nan, 20]})
    df = df.set_index(['a', 'b'])
    expected = DataFrame({'c': [10, 15, np.nan, 20]}, index=[Index([pd.NaT, pd.NaT, pd.NaT, pd.NaT], name='a'), Index(['C1', 'C2', 'C3', 'C4'], name='b')])
    tm.assert_frame_equal(df, expected)