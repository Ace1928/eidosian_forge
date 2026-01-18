import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
def test_getitem_simple(multiindex_dataframe_random_data):
    df = multiindex_dataframe_random_data.T
    expected = df.values[:, 0]
    result = df['foo', 'one'].values
    tm.assert_almost_equal(result, expected)