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
def test_empty_groupby_apply_nonunique_columns():
    df = DataFrame(np.random.default_rng(2).standard_normal((0, 4)))
    df[3] = df[3].astype(np.int64)
    df.columns = [0, 1, 2, 0]
    gb = df.groupby(df[1], group_keys=False)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        res = gb.apply(lambda x: x)
    assert (res.dtypes == df.dtypes).all()