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
def test_consistency_name():
    df = DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'], 'B': ['one', 'one', 'two', 'two', 'two', 'two', 'one', 'two'], 'C': np.random.default_rng(2).standard_normal(8) + 1.0, 'D': np.arange(8)})
    expected = df.groupby(['A']).B.count()
    result = df.B.groupby(df.A).count()
    tm.assert_series_equal(result, expected)