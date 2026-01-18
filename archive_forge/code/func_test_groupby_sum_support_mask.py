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
@pytest.mark.parametrize('func, val', [('sum', 3), ('prod', 2)])
def test_groupby_sum_support_mask(any_numeric_ea_dtype, func, val):
    df = DataFrame({'a': 1, 'b': [1, 2, pd.NA]}, dtype=any_numeric_ea_dtype)
    result = getattr(df.groupby('a'), func)()
    expected = DataFrame({'b': [val]}, index=Index([1], name='a', dtype=any_numeric_ea_dtype), dtype=any_numeric_ea_dtype)
    tm.assert_frame_equal(result, expected)