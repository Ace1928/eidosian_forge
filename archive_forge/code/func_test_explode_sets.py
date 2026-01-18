import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_explode_sets():
    df = pd.DataFrame({'a': [{'x', 'y'}], 'b': [1]}, index=[1])
    result = df.explode(column='a').sort_values(by='a')
    expected = pd.DataFrame({'a': ['x', 'y'], 'b': [1, 1]}, index=[1, 1])
    tm.assert_frame_equal(result, expected)