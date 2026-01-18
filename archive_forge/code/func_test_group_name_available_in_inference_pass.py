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
def test_group_name_available_in_inference_pass():
    df = DataFrame({'a': [0, 0, 1, 1, 2, 2], 'b': np.arange(6)})
    names = []

    def f(group):
        names.append(group.name)
        return group.copy()
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        df.groupby('a', sort=False, group_keys=False).apply(f)
    expected_names = [0, 1, 2]
    assert names == expected_names