import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
def test_groupby_cumprod():
    df = DataFrame({'key': ['b'] * 10, 'value': 2})
    actual = df.groupby('key')['value'].cumprod()
    expected = df.groupby('key', group_keys=False)['value'].apply(lambda x: x.cumprod())
    expected.name = 'value'
    tm.assert_series_equal(actual, expected)
    df = DataFrame({'key': ['b'] * 100, 'value': 2})
    df['value'] = df['value'].astype(float)
    actual = df.groupby('key')['value'].cumprod()
    expected = df.groupby('key', group_keys=False)['value'].apply(lambda x: x.cumprod())
    expected.name = 'value'
    tm.assert_series_equal(actual, expected)