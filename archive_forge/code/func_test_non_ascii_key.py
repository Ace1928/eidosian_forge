import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_non_ascii_key(self):
    testjson = b'[{"\xc3\x9cnic\xc3\xb8de":0,"sub":{"A":1, "B":2}},{"\xc3\x9cnic\xc3\xb8de":1,"sub":{"A":3, "B":4}}]'.decode('utf8')
    testdata = {b'\xc3\x9cnic\xc3\xb8de'.decode('utf8'): [0, 1], 'sub.A': [1, 3], 'sub.B': [2, 4]}
    expected = DataFrame(testdata)
    result = json_normalize(json.loads(testjson))
    tm.assert_frame_equal(result, expected)