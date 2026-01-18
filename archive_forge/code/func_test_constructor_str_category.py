from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('categories', [None, ['a', 'b'], ['a', 'c']])
@pytest.mark.parametrize('ordered', [True, False])
def test_constructor_str_category(self, categories, ordered):
    result = Categorical(['a', 'b'], categories=categories, ordered=ordered, dtype='category')
    expected = Categorical(['a', 'b'], categories=categories, ordered=ordered)
    tm.assert_categorical_equal(result, expected)