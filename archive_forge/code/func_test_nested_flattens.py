import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_nested_flattens(self):
    data = {'flat1': 1, 'dict1': {'c': 1, 'd': 2}, 'nested': {'e': {'c': 1, 'd': 2}, 'd': 2}}
    result = nested_to_record(data)
    expected = {'dict1.c': 1, 'dict1.d': 2, 'flat1': 1, 'nested.d': 2, 'nested.e.c': 1, 'nested.e.d': 2}
    assert result == expected