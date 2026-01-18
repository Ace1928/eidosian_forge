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
def test_merge_inner_join_empty(self):
    df_empty = DataFrame()
    df_a = DataFrame({'a': [1, 2]}, index=[0, 1], dtype='int64')
    result = merge(df_empty, df_a, left_index=True, right_index=True)
    expected = DataFrame({'a': []}, dtype='int64')
    tm.assert_frame_equal(result, expected)