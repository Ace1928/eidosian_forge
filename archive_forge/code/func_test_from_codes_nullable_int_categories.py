from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('validate', [True, False])
def test_from_codes_nullable_int_categories(self, any_numeric_ea_dtype, validate):
    cats = pd.array(range(5), dtype=any_numeric_ea_dtype)
    codes = np.random.default_rng(2).integers(5, size=3)
    dtype = CategoricalDtype(cats)
    arr = Categorical.from_codes(codes, dtype=dtype, validate=validate)
    assert arr.categories.dtype == cats.dtype
    tm.assert_index_equal(arr.categories, Index(cats))