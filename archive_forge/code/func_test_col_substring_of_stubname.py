import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_col_substring_of_stubname(self):
    wide_data = {'node_id': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}, 'A': {0: 0.8, 1: 0.0, 2: 0.25, 3: 1.0, 4: 0.81}, 'PA0': {0: 0.74, 1: 0.56, 2: 0.56, 3: 0.98, 4: 0.6}, 'PA1': {0: 0.77, 1: 0.64, 2: 0.52, 3: 0.98, 4: 0.67}, 'PA3': {0: 0.34, 1: 0.7, 2: 0.52, 3: 0.98, 4: 0.67}}
    wide_df = DataFrame.from_dict(wide_data)
    expected = wide_to_long(wide_df, stubnames=['PA'], i=['node_id', 'A'], j='time')
    result = wide_to_long(wide_df, stubnames='PA', i=['node_id', 'A'], j='time')
    tm.assert_frame_equal(result, expected)