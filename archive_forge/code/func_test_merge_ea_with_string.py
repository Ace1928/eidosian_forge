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
def test_merge_ea_with_string(self, join_type, string_dtype):
    df1 = DataFrame(data={('lvl0', 'lvl1-a'): ['1', '2', '3', '4', None], ('lvl0', 'lvl1-b'): ['4', '5', '6', '7', '8']}, dtype=pd.StringDtype())
    df1_copy = df1.copy()
    df2 = DataFrame(data={('lvl0', 'lvl1-a'): ['1', '2', '3', pd.NA, '5'], ('lvl0', 'lvl1-c'): ['7', '8', '9', pd.NA, '11']}, dtype=string_dtype)
    df2_copy = df2.copy()
    merged = merge(left=df1, right=df2, on=[('lvl0', 'lvl1-a')], how=join_type)
    tm.assert_frame_equal(df1, df1_copy)
    tm.assert_frame_equal(df2, df2_copy)
    expected = Series([np.dtype('O'), pd.StringDtype(), np.dtype('O')], index=MultiIndex.from_tuples([('lvl0', 'lvl1-a'), ('lvl0', 'lvl1-b'), ('lvl0', 'lvl1-c')]))
    tm.assert_series_equal(merged.dtypes, expected)