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
@pytest.mark.parametrize('df1_vals, df2_vals', [([0, 1, 2], Series(['a', 'b', 'a']).astype('category')), ([0.0, 1.0, 2.0], Series(['a', 'b', 'a']).astype('category')), ([0, 1], Series([False, True], dtype=object)), ([0, 1], Series([False, True], dtype=bool))])
def test_merge_incompat_dtypes_are_ok(self, df1_vals, df2_vals):
    df1 = DataFrame({'A': df1_vals})
    df2 = DataFrame({'A': df2_vals})
    result = merge(df1, df2, on=['A'])
    assert is_object_dtype(result.A.dtype)
    result = merge(df2, df1, on=['A'])
    assert is_object_dtype(result.A.dtype) or is_string_dtype(result.A.dtype)