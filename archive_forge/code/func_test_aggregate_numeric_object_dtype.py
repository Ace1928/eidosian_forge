import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_aggregate_numeric_object_dtype():
    df = DataFrame({'key': ['A', 'A', 'B', 'B'], 'col1': list('abcd'), 'col2': [np.nan] * 4}).astype(object)
    result = df.groupby('key').min()
    expected = DataFrame({'key': ['A', 'B'], 'col1': ['a', 'c'], 'col2': [np.nan, np.nan]}).set_index('key').astype(object)
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'key': ['A', 'A', 'B', 'B'], 'col1': list('abcd'), 'col2': range(4)}).astype(object)
    result = df.groupby('key').min()
    expected = DataFrame({'key': ['A', 'B'], 'col1': ['a', 'c'], 'col2': [0, 2]}).set_index('key').astype(object)
    tm.assert_frame_equal(result, expected)