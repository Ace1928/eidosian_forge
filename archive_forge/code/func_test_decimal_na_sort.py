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
@pytest.mark.parametrize('test_series', [True, False])
def test_decimal_na_sort(test_series):
    assert not isinstance(decimal.InvalidOperation, TypeError)
    df = DataFrame({'key': [Decimal(1), Decimal(1), None, None], 'value': [Decimal(2), Decimal(3), Decimal(4), Decimal(5)]})
    gb = df.groupby('key', dropna=False)
    if test_series:
        gb = gb['value']
    result = gb._grouper.result_index
    expected = Index([Decimal(1), None], name='key')
    tm.assert_index_equal(result, expected)