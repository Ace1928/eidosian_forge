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
@pytest.mark.parametrize('dt_args,extra_exp', [({}, {}), ({'utc': True}, {'tz': 'UTC'})])
@pytest.mark.parametrize('wrapper', [None, pd.Series])
def test_convert_pandas_type_to_json_field_datetime(self, dt_args, extra_exp, wrapper):
    data = [1.0, 2.0, 3.0]
    data = pd.to_datetime(data, **dt_args)
    if wrapper is pd.Series:
        data = pd.Series(data, name='values')
    result = convert_pandas_type_to_json_field(data)
    expected = {'name': 'values', 'type': 'datetime'}
    expected.update(extra_exp)
    assert result == expected