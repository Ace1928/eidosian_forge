import datetime
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_add_period_to_array_of_offset(self):
    per = pd.Period('2012-1-1', freq='D')
    pi = pd.period_range('2012-1-1', periods=10, freq='D')
    idx = per - pi
    expected = pd.Index([x + per for x in idx], dtype=object)
    result = idx + per
    tm.assert_index_equal(result, expected)
    result = per + idx
    tm.assert_index_equal(result, expected)