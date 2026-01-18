from collections import (
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal
def test_concat_none_with_timezone_timestamp():
    df1 = DataFrame([{'A': None}])
    df2 = DataFrame([{'A': pd.Timestamp('1990-12-20 00:00:00+00:00')}])
    msg = 'The behavior of DataFrame concatenation with empty or all-NA entries'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = concat([df1, df2], ignore_index=True)
    expected = DataFrame({'A': [None, pd.Timestamp('1990-12-20 00:00:00+00:00')]})
    tm.assert_frame_equal(result, expected)