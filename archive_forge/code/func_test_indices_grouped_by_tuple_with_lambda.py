from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_indices_grouped_by_tuple_with_lambda(self):
    df = DataFrame({'Tuples': ((x, y) for x in [0, 1] for y in np.random.default_rng(2).integers(3, 5, 5))})
    gb = df.groupby('Tuples')
    gb_lambda = df.groupby(lambda x: df.iloc[x, 0])
    expected = gb.indices
    result = gb_lambda.indices
    tm.assert_dict_equal(result, expected)