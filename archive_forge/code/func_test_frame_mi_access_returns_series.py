import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
def test_frame_mi_access_returns_series(dataframe_with_duplicate_index):
    df = dataframe_with_duplicate_index
    expected = Series(['a', 1, 1], index=['h1', 'h3', 'h5'], name='A1')
    result = df['A']['A1']
    tm.assert_series_equal(result, expected)