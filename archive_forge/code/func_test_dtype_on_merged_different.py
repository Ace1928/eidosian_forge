from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
@pytest.mark.parametrize('change', [lambda x: x, lambda x: x.astype(CategoricalDtype(['foo', 'bar', 'bah'])), lambda x: x.astype(CategoricalDtype(ordered=True))])
def test_dtype_on_merged_different(self, change, join_type, left, right, using_infer_string):
    X = change(right.X.astype('object'))
    right = right.assign(X=X)
    assert isinstance(left.X.values.dtype, CategoricalDtype)
    merged = merge(left, right, on='X', how=join_type)
    result = merged.dtypes.sort_index()
    dtype = np.dtype('O') if not using_infer_string else 'string'
    expected = Series([dtype, dtype, np.dtype('int64')], index=['X', 'Y', 'Z'])
    tm.assert_series_equal(result, expected)