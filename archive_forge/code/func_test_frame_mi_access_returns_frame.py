import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
def test_frame_mi_access_returns_frame(dataframe_with_duplicate_index):
    df = dataframe_with_duplicate_index
    expected = DataFrame([['d', 4, 4], ['e', 5, 5]], index=Index(['B2', 'B2'], name='sub'), columns=['h1', 'h3', 'h5']).T
    result = df['A']['B2']
    tm.assert_frame_equal(result, expected)