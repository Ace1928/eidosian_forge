from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_frame_groupby_columns(tsframe):
    mapping = {'A': 0, 'B': 0, 'C': 1, 'D': 1}
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grouped = tsframe.groupby(mapping, axis=1)
    aggregated = grouped.aggregate('mean')
    assert len(aggregated) == len(tsframe)
    assert len(aggregated.columns) == 2
    tf = lambda x: x - x.mean()
    msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        groupedT = tsframe.T.groupby(mapping, axis=0)
    tm.assert_frame_equal(groupedT.transform(tf).T, grouped.transform(tf))
    for k, v in grouped:
        assert len(v.columns) == 2