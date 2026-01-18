import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_mixed_type_suffix(self):
    df = DataFrame({'A': ['X1', 'X2'], 'result_1': [0, 9], 'result_foo': [5.0, 6.0], 'treatment_1': [1.0, 2.0], 'treatment_foo': [3.0, 4.0]})
    expected = DataFrame({'A': ['X1', 'X2', 'X1', 'X2'], 'colname': ['1', '1', 'foo', 'foo'], 'result': [0.0, 9.0, 5.0, 6.0], 'treatment': [1.0, 2.0, 3.0, 4.0]}).set_index(['A', 'colname'])
    result = wide_to_long(df, ['result', 'treatment'], i='A', j='colname', suffix='.+', sep='_')
    tm.assert_frame_equal(result, expected)