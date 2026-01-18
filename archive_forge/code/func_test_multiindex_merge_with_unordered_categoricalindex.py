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
@pytest.mark.parametrize('ordered', [True, False])
def test_multiindex_merge_with_unordered_categoricalindex(self, ordered):
    pcat = CategoricalDtype(categories=['P2', 'P1'], ordered=ordered)
    df1 = DataFrame({'id': ['C', 'C', 'D'], 'p': Categorical(['P2', 'P1', 'P2'], dtype=pcat), 'a': [0, 1, 2]}).set_index(['id', 'p'])
    df2 = DataFrame({'id': ['A', 'C', 'C'], 'p': Categorical(['P2', 'P2', 'P1'], dtype=pcat), 'd1': [10, 11, 12]}).set_index(['id', 'p'])
    result = merge(df1, df2, how='left', left_index=True, right_index=True)
    expected = DataFrame({'id': ['C', 'C', 'D'], 'p': Categorical(['P2', 'P1', 'P2'], dtype=pcat), 'a': [0, 1, 2], 'd1': [11.0, 12.0, np.nan]}).set_index(['id', 'p'])
    tm.assert_frame_equal(result, expected)