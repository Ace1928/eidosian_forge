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
def test_build_int64_series(self, ia):
    s = Series(ia, name='a')
    s.index.name = 'id'
    result = s.to_json(orient='table', date_format='iso')
    result = json.loads(result, object_pairs_hook=OrderedDict)
    assert 'pandas_version' in result['schema']
    result['schema'].pop('pandas_version')
    fields = [{'name': 'id', 'type': 'integer'}, {'name': 'a', 'type': 'integer', 'extDtype': 'Int64'}]
    schema = {'fields': fields, 'primaryKey': ['id']}
    expected = OrderedDict([('schema', schema), ('data', [OrderedDict([('id', 0), ('a', 10)])])])
    assert result == expected