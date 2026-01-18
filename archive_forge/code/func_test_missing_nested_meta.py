import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_missing_nested_meta(self):
    data = {'meta': 'foo', 'nested_meta': None, 'value': [{'rec': 1}, {'rec': 2}]}
    result = json_normalize(data, record_path='value', meta=['meta', ['nested_meta', 'leaf']], errors='ignore')
    ex_data = [[1, 'foo', np.nan], [2, 'foo', np.nan]]
    columns = ['rec', 'meta', 'nested_meta.leaf']
    expected = DataFrame(ex_data, columns=columns).astype({'nested_meta.leaf': object})
    tm.assert_frame_equal(result, expected)
    with pytest.raises(KeyError, match="'leaf' not found"):
        json_normalize(data, record_path='value', meta=['meta', ['nested_meta', 'leaf']], errors='raise')