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
def test_json_ext_dtype_reading_roundtrip(self):
    df = DataFrame({'a': Series([2, NA], dtype='Int64'), 'b': Series([1.5, NA], dtype='Float64'), 'c': Series([True, NA], dtype='boolean')}, index=Index([1, NA], dtype='Int64'))
    expected = df.copy()
    data_json = df.to_json(orient='table', indent=4)
    result = read_json(StringIO(data_json), orient='table')
    tm.assert_frame_equal(result, expected)