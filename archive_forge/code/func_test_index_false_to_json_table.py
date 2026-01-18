import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
@pytest.mark.parametrize('data', [DataFrame([[1, 2], [4, 5]], columns=['a', 'b']), DataFrame([[1, 2], [4, 5]], columns=['a', 'b']).rename_axis('foo'), DataFrame([[1, 2], [4, 5]], columns=['a', 'b'], index=[['a', 'b'], ['c', 'd']]), Series([1, 2, 3], name='A'), Series([1, 2, 3], name='A').rename_axis('foo'), Series([1, 2], name='A', index=[['a', 'b'], ['c', 'd']])])
def test_index_false_to_json_table(self, data):
    result = data.to_json(orient='table', index=False)
    result = json.loads(result)
    expected = {'schema': pd.io.json.build_table_schema(data, index=False), 'data': DataFrame(data).to_dict(orient='records')}
    assert result == expected