import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_melt_ea_columns(self):
    df = DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'}, 'B': {0: 1, 1: 3, 2: 5}, 'C': {0: 2, 1: 4, 2: 6}})
    df.columns = df.columns.astype('string[python]')
    result = df.melt(id_vars=['A'], value_vars=['B'])
    expected = DataFrame({'A': list('abc'), 'variable': pd.Series(['B'] * 3, dtype='string[python]'), 'value': [1, 3, 5]})
    tm.assert_frame_equal(result, expected)