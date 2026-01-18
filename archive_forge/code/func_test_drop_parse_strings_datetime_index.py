import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_drop_parse_strings_datetime_index(self):
    df = DataFrame({'a': [1, 2], 'b': [1, 2]}, index=[Timestamp('2000-01-03'), Timestamp('2000-01-04')])
    result = df.drop('2000-01-03', axis=0)
    expected = DataFrame({'a': [2], 'b': [2]}, index=[Timestamp('2000-01-04')])
    tm.assert_frame_equal(result, expected)