import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
@pytest.fixture
def max_level_test_input_data():
    """
    input data to test json_normalize with max_level param
    """
    return [{'CreatedBy': {'Name': 'User001'}, 'Lookup': {'TextField': 'Some text', 'UserField': {'Id': 'ID001', 'Name': 'Name001'}}, 'Image': {'a': 'b'}}]