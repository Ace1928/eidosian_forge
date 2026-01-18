import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
@pytest.mark.parametrize('data, record_path, exception_type', [([{'a': 0}, {'a': 1}], None, None), ({'a': [{'a': 0}, {'a': 1}]}, 'a', None), ('{"a": [{"a": 0}, {"a": 1}]}', None, NotImplementedError), (None, None, NotImplementedError)])
def test_accepted_input(self, data, record_path, exception_type):
    if exception_type is not None:
        with pytest.raises(exception_type, match=''):
            json_normalize(data, record_path=record_path)
    else:
        result = json_normalize(data, record_path=record_path)
        expected = DataFrame([0, 1], columns=['a'])
        tm.assert_frame_equal(result, expected)