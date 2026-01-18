import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_fields_list_type_normalize(self):
    parse_metadata_fields_list_type = [{'values': [1, 2, 3], 'metadata': {'listdata': [1, 2]}}]
    result = json_normalize(parse_metadata_fields_list_type, record_path=['values'], meta=[['metadata', 'listdata']])
    expected = DataFrame({0: [1, 2, 3], 'metadata.listdata': [[1, 2], [1, 2], [1, 2]]})
    tm.assert_frame_equal(result, expected)