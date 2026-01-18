import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_meta_non_iterable(self):
    data = '[{"id": 99, "data": [{"one": 1, "two": 2}]}]'
    result = json_normalize(json.loads(data), record_path=['data'], meta=['id'])
    expected = DataFrame({'one': [1], 'two': [2], 'id': np.array([99], dtype=object)})
    tm.assert_frame_equal(result, expected)