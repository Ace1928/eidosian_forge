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
@pytest.mark.parametrize('bug_var', [1, 'a'])
def test_groupby_sum_on_nan_should_return_nan(bug_var):
    df = DataFrame({'A': [bug_var, bug_var, bug_var, np.nan]})
    dfgb = df.groupby(lambda x: x)
    result = dfgb.sum(min_count=1)
    expected_df = DataFrame([bug_var, bug_var, bug_var, None], columns=['A'])
    tm.assert_frame_equal(result, expected_df)