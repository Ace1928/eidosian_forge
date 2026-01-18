from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_multiindex_passthru(self):
    df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df.columns = MultiIndex.from_tuples([(0, 1), (1, 1), (2, 1)])
    depr_msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        gb = df.groupby(axis=1, level=[0, 1])
    result = gb.first()
    tm.assert_frame_equal(result, df)