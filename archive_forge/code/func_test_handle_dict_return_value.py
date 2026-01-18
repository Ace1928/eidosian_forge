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
def test_handle_dict_return_value(df):

    def f(group):
        return {'max': group.max(), 'min': group.min()}

    def g(group):
        return Series({'max': group.max(), 'min': group.min()})
    result = df.groupby('A')['C'].apply(f)
    expected = df.groupby('A')['C'].apply(g)
    assert isinstance(result, Series)
    tm.assert_series_equal(result, expected)