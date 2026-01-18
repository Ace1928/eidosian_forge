from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ordered', [True, False])
def test_constructor_with_dtype(self, ordered):
    categories = ['b', 'a', 'c']
    dtype = CategoricalDtype(categories, ordered=ordered)
    result = Categorical(['a', 'b', 'a', 'c'], dtype=dtype)
    expected = Categorical(['a', 'b', 'a', 'c'], categories=categories, ordered=ordered)
    tm.assert_categorical_equal(result, expected)
    assert result.ordered is ordered