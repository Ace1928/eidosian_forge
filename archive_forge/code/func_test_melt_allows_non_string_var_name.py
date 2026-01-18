import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_melt_allows_non_string_var_name(self):
    df = DataFrame(data={'a': [1, 2, 3], 'b': [4, 5, 6]}, index=['11', '22', '33'])
    result = df.melt(id_vars=['a'], var_name=0, value_name=1)
    expected = DataFrame({'a': [1, 2, 3], 0: ['b'] * 3, 1: [4, 5, 6]})
    tm.assert_frame_equal(result, expected)