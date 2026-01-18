import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
def test_bins_from_interval_index():
    c = cut(range(5), 3)
    expected = c
    result = cut(range(5), bins=expected.categories)
    tm.assert_categorical_equal(result, expected)
    expected = Categorical.from_codes(np.append(c.codes, -1), categories=c.categories, ordered=True)
    result = cut(range(6), bins=expected.categories)
    tm.assert_categorical_equal(result, expected)