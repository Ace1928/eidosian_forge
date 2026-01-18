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
def test_json_ext_dtype_reading(self):
    data_json = '{\n            "schema":{\n                "fields":[\n                    {\n                        "name":"a",\n                        "type":"integer",\n                        "extDtype":"Int64"\n                    }\n                ],\n            },\n            "data":[\n                {\n                    "a":2\n                },\n                {\n                    "a":null\n                }\n            ]\n        }'
    result = read_json(StringIO(data_json), orient='table')
    expected = DataFrame({'a': Series([2, NA], dtype='Int64')})
    tm.assert_frame_equal(result, expected)