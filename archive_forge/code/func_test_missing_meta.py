import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_missing_meta(self, missing_metadata):
    result = json_normalize(data=missing_metadata, record_path='addresses', meta='name', errors='ignore')
    ex_data = [[9562, 'Morris St.', 'Massillon', 'OH', 44646, 'Alice'], [8449, 'Spring St.', 'Elizabethton', 'TN', 37643, np.nan]]
    columns = ['number', 'street', 'city', 'state', 'zip', 'name']
    expected = DataFrame(ex_data, columns=columns)
    tm.assert_frame_equal(result, expected)