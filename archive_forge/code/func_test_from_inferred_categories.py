from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', [None, 'category'])
def test_from_inferred_categories(self, dtype):
    cats = ['a', 'b']
    codes = np.array([0, 0, 1, 1], dtype='i8')
    result = Categorical._from_inferred_categories(cats, codes, dtype)
    expected = Categorical.from_codes(codes, cats)
    tm.assert_categorical_equal(result, expected)