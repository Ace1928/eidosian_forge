import numpy as np
import pytest
from pandas.core.dtypes.common import is_bool_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base
@pytest.mark.parametrize('as_index', [True, False])
def test_groupby_extension_agg(self, as_index, data_for_grouping):
    df = pd.DataFrame({'A': [1, 1, 2, 2, 3, 3, 1], 'B': data_for_grouping})
    result = df.groupby('B', as_index=as_index).A.mean()
    _, uniques = pd.factorize(data_for_grouping, sort=True)
    if as_index:
        index = pd.Index(uniques, name='B')
        expected = pd.Series([3.0, 1.0], index=index, name='A')
        self.assert_series_equal(result, expected)
    else:
        expected = pd.DataFrame({'B': uniques, 'A': [3.0, 1.0]})
        self.assert_frame_equal(result, expected)