import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize(('data', 'f', 'expected'), (([1, 1, np.nan], pd.isna, CategoricalIndex([False, False, np.nan])), ([1, 2, np.nan], pd.isna, Index([False, False, np.nan])), ([1, 1, np.nan], {1: False}, CategoricalIndex([False, False, np.nan])), ([1, 2, np.nan], {1: False, 2: False}, Index([False, False, np.nan])), ([1, 1, np.nan], Series([False, False]), CategoricalIndex([False, False, np.nan])), ([1, 2, np.nan], Series([False, False, False]), Index([False, False, np.nan]))))
def test_map_with_nan_ignore(data, f, expected):
    values = CategoricalIndex(data)
    result = values.map(f, na_action='ignore')
    tm.assert_index_equal(result, expected)