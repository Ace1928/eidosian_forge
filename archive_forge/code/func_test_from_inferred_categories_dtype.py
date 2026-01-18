from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_inferred_categories_dtype(self):
    cats = ['a', 'b', 'd']
    codes = np.array([0, 1, 0, 2], dtype='i8')
    dtype = CategoricalDtype(['c', 'b', 'a'], ordered=True)
    result = Categorical._from_inferred_categories(cats, codes, dtype)
    expected = Categorical(['a', 'b', 'a', 'd'], categories=['c', 'b', 'a'], ordered=True)
    tm.assert_categorical_equal(result, expected)