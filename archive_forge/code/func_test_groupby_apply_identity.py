import numpy as np
import pytest
from pandas.core.dtypes.common import is_bool_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base
def test_groupby_apply_identity(self, data_for_grouping):
    df = pd.DataFrame({'A': [1, 1, 2, 2, 3, 3, 1], 'B': data_for_grouping})
    result = df.groupby('A').B.apply(lambda x: x.array)
    expected = pd.Series([df.B.iloc[[0, 1, 6]].array, df.B.iloc[[2, 3]].array, df.B.iloc[[4, 5]].array], index=pd.Index([1, 2, 3], name='A'), name='B')
    self.assert_series_equal(result, expected)