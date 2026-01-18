import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_multiindex_indexer_none(self):
    attributes = ['Attribute' + str(i) for i in range(1)]
    attribute_values = ['Value' + str(i) for i in range(5)]
    index = MultiIndex.from_product([attributes, attribute_values])
    df = 0.1 * np.random.default_rng(2).standard_normal((10, 1 * 5)) + 0.5
    df = DataFrame(df, columns=index)
    result = df[attributes]
    tm.assert_frame_equal(result, df)
    df = DataFrame(np.arange(12).reshape(-1, 1), index=MultiIndex.from_product([[1, 2, 3, 4], [1, 2, 3]]))
    expected = df.loc[([1, 2],), :]
    result = df.loc[[1, 2]]
    tm.assert_frame_equal(result, expected)