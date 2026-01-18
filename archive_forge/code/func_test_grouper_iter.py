from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_grouper_iter(self, df):
    gb = df.groupby('A')
    msg = 'DataFrameGroupBy.grouper is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grouper = gb.grouper
    result = sorted(grouper)
    expected = ['bar', 'foo']
    assert result == expected