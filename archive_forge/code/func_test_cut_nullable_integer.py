import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
@pytest.mark.parametrize('bins', [3, [0, 5, 15]])
@pytest.mark.parametrize('right', [True, False])
@pytest.mark.parametrize('include_lowest', [True, False])
def test_cut_nullable_integer(bins, right, include_lowest):
    a = np.random.default_rng(2).integers(0, 10, size=50).astype(float)
    a[::2] = np.nan
    result = cut(pd.array(a, dtype='Int64'), bins, right=right, include_lowest=include_lowest)
    expected = cut(a, bins, right=right, include_lowest=include_lowest)
    tm.assert_categorical_equal(result, expected)