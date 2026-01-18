from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_frame_yield_constant(df):
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby(['A', 'B']).apply(len)
    assert isinstance(result, Series)
    assert result.name is None
    result = df.groupby(['A', 'B'])[['C', 'D']].apply(len)
    assert isinstance(result, Series)
    assert result.name is None