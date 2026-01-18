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
@pytest.mark.parametrize('func', ['sum', 'any', 'shift'])
def test_groupby_column_index_name_lost(func):
    expected = Index(['a'], name='idx')
    df = DataFrame([[1]], columns=expected)
    df_grouped = df.groupby([1])
    result = getattr(df_grouped, func)().columns
    tm.assert_index_equal(result, expected)