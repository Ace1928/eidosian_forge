import numpy as np
import pytest
from pandas.core.dtypes.common import is_bool_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base
def test_groupby_extension_no_sort(self, data_for_grouping):
    df = pd.DataFrame({'A': [1, 1, 2, 2, 3, 3, 1], 'B': data_for_grouping})
    result = df.groupby('B', sort=False).A.mean()
    _, index = pd.factorize(data_for_grouping, sort=False)
    index = pd.Index(index, name='B')
    expected = pd.Series([1.0, 3.0], index=index, name='A')
    self.assert_series_equal(result, expected)