from collections import OrderedDict
import datetime as dt
import decimal
from io import StringIO
import json
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.string_ import StringDtype
from pandas.core.series import Series
from pandas.tests.extension.date import (
from pandas.tests.extension.decimal.array import (
from pandas.io.json._table_schema import (
def test_build_table_schema(self):
    df = DataFrame({'A': DateArray([dt.date(2021, 10, 10)]), 'B': DecimalArray([decimal.Decimal(10)]), 'C': array(['pandas'], dtype='string'), 'D': array([10], dtype='Int64')})
    result = build_table_schema(df, version=False)
    expected = {'fields': [{'name': 'index', 'type': 'integer'}, {'name': 'A', 'type': 'any', 'extDtype': 'DateDtype'}, {'name': 'B', 'type': 'number', 'extDtype': 'decimal'}, {'name': 'C', 'type': 'any', 'extDtype': 'string'}, {'name': 'D', 'type': 'integer', 'extDtype': 'Int64'}], 'primaryKey': ['index']}
    assert result == expected
    result = build_table_schema(df)
    assert 'pandas_version' in result