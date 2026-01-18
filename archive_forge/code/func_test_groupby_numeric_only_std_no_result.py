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
@pytest.mark.parametrize('numeric_only', [True, False])
def test_groupby_numeric_only_std_no_result(numeric_only):
    dicts_non_numeric = [{'a': 'foo', 'b': 'bar'}, {'a': 'car', 'b': 'dar'}]
    df = DataFrame(dicts_non_numeric)
    dfgb = df.groupby('a', as_index=False, sort=False)
    if numeric_only:
        result = dfgb.std(numeric_only=True)
        expected_df = DataFrame(['foo', 'car'], columns=['a'])
        tm.assert_frame_equal(result, expected_df)
    else:
        with pytest.raises(ValueError, match="could not convert string to float: 'bar'"):
            dfgb.std(numeric_only=numeric_only)