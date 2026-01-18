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
@pytest.mark.parametrize('sparse', [True, False])
def test_get_dummies_dont_sparsify_all_columns(self, sparse):
    df = DataFrame.from_dict({'GDP': [1, 2], 'Nation': ['AB', 'CD']})
    df = get_dummies(df, columns=['Nation'], sparse=sparse)
    df2 = df.reindex(columns=['GDP'])
    tm.assert_frame_equal(df[['GDP']], df2)