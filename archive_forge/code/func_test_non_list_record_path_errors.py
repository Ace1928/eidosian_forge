import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
@pytest.mark.parametrize('value', ['false', 'true', '{}', '1', '"text"'])
def test_non_list_record_path_errors(self, value):
    parsed_value = json.loads(value)
    test_input = {'state': 'Texas', 'info': parsed_value}
    test_path = 'info'
    msg = f'{test_input} has non list value {parsed_value} for path {test_path}. Must be list or null.'
    with pytest.raises(TypeError, match=msg):
        json_normalize([test_input], record_path=[test_path])