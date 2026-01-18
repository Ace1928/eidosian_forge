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
def test_merge_index_singlekey_inner(self):
    left = DataFrame({'key': ['a', 'b', 'c', 'd', 'e', 'e', 'a'], 'v1': np.random.default_rng(2).standard_normal(7)})
    right = DataFrame({'v2': np.random.default_rng(2).standard_normal(4)}, index=['d', 'b', 'c', 'a'])
    result = merge(left, right, left_on='key', right_index=True, how='inner')
    expected = left.join(right, on='key').loc[result.index]
    tm.assert_frame_equal(result, expected)
    result = merge(right, left, right_on='key', left_index=True, how='inner')
    expected = left.join(right, on='key').loc[result.index]
    tm.assert_frame_equal(result, expected.loc[:, result.columns])