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
@pytest.mark.parametrize('kind', [pd.Categorical, pd.CategoricalIndex])
@pytest.mark.parametrize('ordered', [True, False])
def test_convert_pandas_type_to_json_field_categorical(self, kind, ordered):
    data = ['a', 'b', 'c']
    if kind is pd.Categorical:
        arr = pd.Series(kind(data, ordered=ordered), name='cats')
    elif kind is pd.CategoricalIndex:
        arr = kind(data, ordered=ordered, name='cats')
    result = convert_pandas_type_to_json_field(arr)
    expected = {'name': 'cats', 'type': 'any', 'constraints': {'enum': data}, 'ordered': ordered}
    assert result == expected