import re
import unicodedata
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('ordered', [True, False])
def test_dataframe_dummies_preserve_categorical_dtype(self, dtype, ordered):
    cat = Categorical(list('xy'), categories=list('xyz'), ordered=ordered)
    result = get_dummies(cat, dtype=dtype)
    data = np.array([[1, 0, 0], [0, 1, 0]], dtype=self.effective_dtype(dtype))
    cols = CategoricalIndex(cat.categories, categories=cat.categories, ordered=ordered)
    expected = DataFrame(data, columns=cols, dtype=self.effective_dtype(dtype))
    tm.assert_frame_equal(result, expected)