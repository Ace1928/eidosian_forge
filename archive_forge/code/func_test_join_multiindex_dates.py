from datetime import datetime
import numpy as np
import pytest
from pandas.errors import MergeError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
def test_join_multiindex_dates(self):
    date = pd.Timestamp(2000, 1, 1).date()
    df1_index = MultiIndex.from_tuples([(0, date)], names=['index_0', 'date'])
    df1 = DataFrame({'col1': [0]}, index=df1_index)
    df2_index = MultiIndex.from_tuples([(0, date)], names=['index_0', 'date'])
    df2 = DataFrame({'col2': [0]}, index=df2_index)
    df3_index = MultiIndex.from_tuples([(0, date)], names=['index_0', 'date'])
    df3 = DataFrame({'col3': [0]}, index=df3_index)
    result = df1.join([df2, df3])
    expected_index = MultiIndex.from_tuples([(0, date)], names=['index_0', 'date'])
    expected = DataFrame({'col1': [0], 'col2': [0], 'col3': [0]}, index=expected_index)
    tm.assert_equal(result, expected)