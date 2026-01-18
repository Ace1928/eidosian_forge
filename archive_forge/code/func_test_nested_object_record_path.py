import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_nested_object_record_path(self):
    data = {'state': 'Florida', 'info': {'governor': 'Rick Scott', 'counties': [{'name': 'Dade', 'population': 12345}, {'name': 'Broward', 'population': 40000}, {'name': 'Palm Beach', 'population': 60000}]}}
    result = json_normalize(data, record_path=['info', 'counties'])
    expected = DataFrame([['Dade', 12345], ['Broward', 40000], ['Palm Beach', 60000]], columns=['name', 'population'])
    tm.assert_frame_equal(result, expected)