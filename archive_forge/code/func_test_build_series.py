from collections import OrderedDict
from io import StringIO
import json
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.json._table_schema import (
def test_build_series(self):
    s = pd.Series([1, 2], name='a')
    s.index.name = 'id'
    result = s.to_json(orient='table', date_format='iso')
    result = json.loads(result, object_pairs_hook=OrderedDict)
    assert 'pandas_version' in result['schema']
    result['schema'].pop('pandas_version')
    fields = [{'name': 'id', 'type': 'integer'}, {'name': 'a', 'type': 'integer'}]
    schema = {'fields': fields, 'primaryKey': ['id']}
    expected = OrderedDict([('schema', schema), ('data', [OrderedDict([('id', 0), ('a', 1)]), OrderedDict([('id', 1), ('a', 2)])])])
    assert result == expected