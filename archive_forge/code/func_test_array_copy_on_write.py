from __future__ import annotations
import decimal
import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
def test_array_copy_on_write(using_copy_on_write):
    df = pd.DataFrame({'a': [decimal.Decimal(2), decimal.Decimal(3)]}, dtype='object')
    df2 = df.astype(DecimalDtype())
    df.iloc[0, 0] = 0
    if using_copy_on_write:
        expected = pd.DataFrame({'a': [decimal.Decimal(2), decimal.Decimal(3)]}, dtype=DecimalDtype())
        tm.assert_equal(df2.values, expected.values)