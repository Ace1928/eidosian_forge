import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_ignore_index():
    df = pd.DataFrame({'id': range(0, 20, 10), 'values': [list('ab'), list('cd')]})
    result = df.explode('values', ignore_index=True)
    expected = pd.DataFrame({'id': [0, 0, 10, 10], 'values': list('abcd')}, index=[0, 1, 2, 3])
    tm.assert_frame_equal(result, expected)