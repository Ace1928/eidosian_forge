import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_drop_inplace_no_leftover_column_reference(self):
    df = DataFrame({'a': [1, 2, 3]}, columns=Index(['a'], dtype='object'))
    a = df.a
    df.drop(['a'], axis=1, inplace=True)
    tm.assert_index_equal(df.columns, Index([], dtype='object'))
    a -= a.mean()
    tm.assert_index_equal(df.columns, Index([], dtype='object'))